#include "connected_layer.h"
#include "batchnorm_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t get_connected_workspace_size(layer l) {
#ifdef CUDNN
    return get_convolutional_workspace_size(l);
    /*
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dweightDesc,
            l.bf_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dsrcTensorDesc,
            l.bd_algo,
            &s));
        if (s > most) most = s;
        return most;
    }
    */
#endif
    return 0;
}

connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize) {
    int total_batch = batch * steps;
    int i;
    connected_layer l = { (LAYER_TYPE)0 };
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.n = l.out_c;
    l.size = 1;
    l.stride = l.stride_x = l.stride_y = 1;
    l.pad = 0;
    l.activation = activation;
    l.learning_rate_scale = 1;
    l.groups = 1;
    l.dilation = 1;

    l.output = (float*)xcalloc(total_batch * outputs, sizeof(float));
    l.delta = (float*)xcalloc(total_batch * outputs, sizeof(float));

    l.weight_updates = (float*)xcalloc(inputs * outputs, sizeof(float));
    l.bias_updates = (float*)xcalloc(outputs, sizeof(float));

    l.weights = (float*)xcalloc(outputs * inputs, sizeof(float));
    l.biases = (float*)xcalloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2.f / inputs);
    for (i = 0; i < outputs * inputs; ++i) {
        l.weights[i] = scale * rand_uniform(-1, 1);
    }

    for (i = 0; i < outputs; ++i) {
        l.biases[i] = 0;
    }

    if (batch_normalize) {
        l.scales = (float*)xcalloc(outputs, sizeof(float));
        l.scale_updates = (float*)xcalloc(outputs, sizeof(float));
        for (i = 0; i < outputs; ++i) {
            l.scales[i] = 1;
        }

        l.mean = (float*)xcalloc(outputs, sizeof(float));
        l.mean_delta = (float*)xcalloc(outputs, sizeof(float));
        l.variance = (float*)xcalloc(outputs, sizeof(float));
        l.variance_delta = (float*)xcalloc(outputs, sizeof(float));

        l.rolling_mean = (float*)xcalloc(outputs, sizeof(float));
        l.rolling_variance = (float*)xcalloc(outputs, sizeof(float));

        l.x = (float*)xcalloc(total_batch * outputs, sizeof(float));
        l.x_norm = (float*)xcalloc(total_batch * outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs * inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs * inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs * total_batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs * total_batch);
    if (batch_normalize) {
        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.x_gpu = cuda_make_array(l.output, total_batch * outputs);
        l.x_norm_gpu = cuda_make_array(l.output, total_batch * outputs);
    }
#ifdef CUDNN
    create_convolutional_cudnn_tensors(&l);
    cudnn_convolutional_setup(&l, cudnn_fastest, 0);   // cudnn_fastest, cudnn_smallest
    l.workspace_size = get_connected_workspace_size(l);
#endif  // CUDNN
#endif  // GPU
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

/// <summary>
/// 更新  全连接层  参数
/// </summary>
/// <param name="l"></param>
/// <param name="batch"></param>
/// <param name="learning_rate"></param>
/// <param name="momentum"></param>
/// <param name="decay"></param>
void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay) {
    //更新 偏移
    // l.biases += learning_rate/batch * l.bias_updates
    axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    //动量抹平
    // l.bias_updates *= momentum
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    //BN层更新  -->  l.scales
    if (l.batch_normalize) {
        axpy_cpu(l.outputs, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    //更新 权重
    // 
    axpy_cpu(l.inputs * l.outputs, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs * l.outputs, momentum, l.weight_updates, 1);
}

/// <summary>
/// 全连接层 前向传播
/// </summary>
/// <param name="l">本层</param>
/// <param name="state">前一层</param>
void forward_connected_layer(connected_layer l, network_state state) {
    int i;
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float* a = state.input;
    float* b = l.weights;
    float* c = l.output;

    //全连接运算 --> 滑动线性一维卷积
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if (l.batch_normalize) {
        //训练
        if (state.train) {
            // 计算均值  方差  将输出变换为  标准正态分布 并附带一点线性偏移
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);

            // 对方差与均值 做一阶低通滤波 固定参数
            // l.rolling_mean *= .95f
            scal_cpu(l.outputs, .95f, l.rolling_mean, 1);
            // l.rolling_mean += .05f*l.mean
            axpy_cpu(l.outputs, .05f, l.mean, 1, l.rolling_mean, 1);
            // l.rolling_variance *= .95f
            scal_cpu(l.outputs, .95f, l.rolling_variance, 1);
            // l.rolling_variance += .05f*l.variance
            axpy_cpu(l.outputs, .05f, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
            // 输出标准化 --> 正态分布
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
            copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1);
        } else {
            // 输出标准化 --> 正态分布
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        // 对输出缩放
        // l.output *= l.scales
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
    for (i = 0; i < l.batch; ++i) {
        //  *(l.output + i*l.outputs) *=  l.biases*l.output
        //  i*l.outputs 代表运算到第几批 --> 地址
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output + i * l.outputs, 1);
    }

    //激活        l.outputs*l.batch 总数据长度
    activate_array(l.output, l.outputs * l.batch, l.activation);
}



/// <summary>
/// 全连接层反向传播
/// </summary>
/// <param name="l">本层</param>
/// <param name="state">前一层</param>
void backward_connected_layer(connected_layer l, network_state state) {
    int i;
    // 计算激活函数的梯度
    // 即：激活函数的导数  并写入 l.delta
    // 划重点！！！  l.delta  中存放的是 激活函数的导数值  梯度的一部分
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    // delta 应当为损失 误差 
    for (i = 0; i < l.batch; ++i) {
        // 更新不同批的偏移 bias_updates += *(delta + i*outputs)
        axpy_cpu(l.outputs, 1, l.delta + i * l.outputs, 1, l.bias_updates, 1);
    }

    //BN 层更新   梯度标准化
    if (l.batch_normalize) {
        // scale_updates += sigma(delta * x_norm)
        backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);

        // output *= scales
        scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);
        // delta 均值
        mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
        // delta 方差
        variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
        // delta 正态标准化
        normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float* a = l.delta;                 // 梯度
    float* b = state.input;             // 上一层输入信息 x
    float* c = l.weight_updates;        // c 项设定为权重  --> 更新权重 (由 B 与 C 滑动卷积更新)

    // 权重更新 w += learn * gradient * x --> 推算 l.delta 为梯度的含义

    // 更新本层参数 C += A*B*1
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;                    // c 项设定为 上一层梯度  则更新前一层梯度信息

    //更新前一层梯度
    if (c) gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);

    /// 看的太简单
    /// 1. 没有相关知识，不存在认识了解，不存在经验
    /// 2. 思维方式问题(面对未知或者在极少了解的情况的态度-- > 模糊, 每次自己思考的仅仅是一种可能, 应当深入了解, 基本信息, 再逆向思考每一步的充分条件)

    /// 认真思考 不同领域关于学习或其范围的定义
    /// 1. 互联网领域学习要求：
    ///  1 > 代码规范-- > 统一规范方便合作，提高效率
    ///  2 > 代码实现分块、职能分割思想-- > 方便复用移植，提高效率
    ///  3 > 需求分析的思路-- > 提高开发成功率
    /// 2. 嵌入式单片机领域关于学习或者其范围的定义
    ///  1 > 单片机使用，嵌入式驱动开发-- > 基本
    ///  2 > 代码规范-- > 统一规范方便合作，提高效率
    ///  3 > 代码实现分块、职能分割思想-- > 方便复用移植，提高效率
    ///  4 > 需求分析的思路-- > 提高开发成功率
    /// 3. 算法测试开发
    ///  1 > 基本 测试程序开发-- > 基本
    ///  2 > 代码规范-- > 统一规范方便合作，提高效率
    ///  3 > 代码实现分块、职能分割思想-- > 方便复用移植，提高效率
    ///  4 > 需求分析的思路-- > 提高开发成功率
    /// 4. 算法开发
    ///  1 > 研究多种不同的算法实现
    ///  2 > 分析项目需求
    ///  3 > 设计项目需求的算法实现

    /// 学长经常谈到的学习并不是学习本身的含义探讨
    /// 而是形成一种特定的习惯, 不包含新的知识：
    /// 1. 一套代码规范(编程学习) --> 为了企业中员工能合作、提高个体与团队的开发效率(核心)
    /// 2. 一套需求分析思路
    /// 3. 一套问题解决策略
    /// 
    /// 企业家的学习本质上是为了提高企业运行效率
    /// 
    /// 认知吧  感觉是知识层面的断层 与 没有经验 所致现在的处境

}

//非标准的连接层 
void denormalize_connected_layer(layer l) {
    int i, j;
    for (i = 0; i < l.outputs; ++i) {
        float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .000001f);
        for (j = 0; j < l.inputs; ++j) {
            l.weights[i * l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

//统计 全连接层 信息打印
void statistics_connected_layer(layer l) {
    if (l.batch_normalize) {
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
        printf("Rolling Mean ");
        print_statistics(l.rolling_mean, l.outputs);
        printf("Rolling Variance ");
        print_statistics(l.rolling_variance, l.outputs);
        */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(connected_layer l) {
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize) {
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void push_connected_layer(connected_layer l) {
    cuda_push_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize) {
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void update_connected_layer_gpu(connected_layer l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale) {
    float learning_rate = learning_rate_init * l.learning_rate_scale;

    // Loss scale for Mixed-Precision on Tensor-Cores
    if (loss_scale != 1.0) {
        scal_ongpu(l.inputs * l.outputs, 1.0 / loss_scale, l.weight_updates_gpu, 1);
        scal_ongpu(l.outputs, 1.0 / loss_scale, l.bias_updates_gpu, 1);
        scal_ongpu(l.outputs, 1.0 / loss_scale, l.scale_updates_gpu, 1);
    }

    axpy_ongpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.outputs, momentum, l.bias_updates_gpu, 1);

    if (l.batch_normalize) {
        axpy_ongpu(l.outputs, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_ongpu(l.outputs, momentum, l.scale_updates_gpu, 1);
    }

    axpy_ongpu(l.inputs * l.outputs, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(l.inputs * l.outputs, momentum, l.weight_updates_gpu, 1);
}

void forward_connected_layer_gpu(connected_layer l, network_state state) {
    fill_ongpu(l.outputs * l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float* a = state.input;
    float* b = l.weights_gpu;
    float* c = l.output_gpu;
#ifdef CUDNN
    float one = 1;    // alpha[0], beta[0]
    float alpha = 1, beta = 0;

    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
        &alpha, //&one,
        l.srcTensorDesc,
        state.input,
        l.weightDesc,
        l.weights_gpu,
        l.convDesc,
        l.fw_algo,
        state.workspace,
        l.workspace_size,
        &beta,  //&one,
        l.dstTensorDesc,
        l.output_gpu));
#else // CUDNN
    gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
#endif // CUDNN

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    //for(i = 0; i < l.batch; ++i) axpy_ongpu(l.outputs, 1, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

void backward_connected_layer_gpu(connected_layer l, network_state state) {
    int i;
    constrain_ongpu(l.outputs * l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    for (i = 0; i < l.batch; ++i) {
        axpy_ongpu(l.outputs, 1, l.delta_gpu + i * l.outputs, 1, l.bias_updates_gpu, 1);
    }

    if (l.batch_normalize) {
        backward_batchnorm_layer_gpu(l, state);
    }

#ifdef CUDNN_DISABLED
    float one = 1;
    // calculate conv weight updates
    // if used: beta=1 then loss decreases faster
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
        &one,
        l.srcTensorDesc,
        state.input,
        l.ddstTensorDesc,
        l.delta_gpu,
        l.convDesc,
        l.bf_algo,
        state.workspace,
        l.workspace_size,
        &one,
        l.dweightDesc,
        l.weight_updates_gpu));

    if (state.delta) {
        // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
        // calculate delta for the next layer

        CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
            &one,
            l.weightDesc,
            l.weights_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bd_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dsrcTensorDesc,
            state.delta));
    }
#else // CUDNN

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float* a = l.delta_gpu;
    float* b = state.input;
    float* c = l.weight_updates_gpu;

    gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = state.delta;

    if (c) gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
#endif // CUDNN
}
#endif
