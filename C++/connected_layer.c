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
/// ����  ȫ���Ӳ�  ����
/// </summary>
/// <param name="l"></param>
/// <param name="batch"></param>
/// <param name="learning_rate"></param>
/// <param name="momentum"></param>
/// <param name="decay"></param>
void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay) {
    //���� ƫ��
    // l.biases += learning_rate/batch * l.bias_updates
    axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    //����Ĩƽ
    // l.bias_updates *= momentum
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    //BN�����  -->  l.scales
    if (l.batch_normalize) {
        axpy_cpu(l.outputs, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    //���� Ȩ��
    // 
    axpy_cpu(l.inputs * l.outputs, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs * l.outputs, momentum, l.weight_updates, 1);
}

/// <summary>
/// ȫ���Ӳ� ǰ�򴫲�
/// </summary>
/// <param name="l">����</param>
/// <param name="state">ǰһ��</param>
void forward_connected_layer(connected_layer l, network_state state) {
    int i;
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float* a = state.input;
    float* b = l.weights;
    float* c = l.output;

    //ȫ�������� --> ��������һά���
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if (l.batch_normalize) {
        //ѵ��
        if (state.train) {
            // �����ֵ  ����  ������任Ϊ  ��׼��̬�ֲ� ������һ������ƫ��
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);

            // �Է������ֵ ��һ�׵�ͨ�˲� �̶�����
            // l.rolling_mean *= .95f
            scal_cpu(l.outputs, .95f, l.rolling_mean, 1);
            // l.rolling_mean += .05f*l.mean
            axpy_cpu(l.outputs, .05f, l.mean, 1, l.rolling_mean, 1);
            // l.rolling_variance *= .95f
            scal_cpu(l.outputs, .95f, l.rolling_variance, 1);
            // l.rolling_variance += .05f*l.variance
            axpy_cpu(l.outputs, .05f, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
            // �����׼�� --> ��̬�ֲ�
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
            copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1);
        } else {
            // �����׼�� --> ��̬�ֲ�
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        // ���������
        // l.output *= l.scales
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
    for (i = 0; i < l.batch; ++i) {
        //  *(l.output + i*l.outputs) *=  l.biases*l.output
        //  i*l.outputs �������㵽�ڼ��� --> ��ַ
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output + i * l.outputs, 1);
    }

    //����        l.outputs*l.batch �����ݳ���
    activate_array(l.output, l.outputs * l.batch, l.activation);
}



/// <summary>
/// ȫ���Ӳ㷴�򴫲�
/// </summary>
/// <param name="l">����</param>
/// <param name="state">ǰһ��</param>
void backward_connected_layer(connected_layer l, network_state state) {
    int i;
    // ���㼤������ݶ�
    // ����������ĵ���  ��д�� l.delta
    // ���ص㣡����  l.delta  �д�ŵ��� ������ĵ���ֵ  �ݶȵ�һ����
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    // delta Ӧ��Ϊ��ʧ ��� 
    for (i = 0; i < l.batch; ++i) {
        // ���²�ͬ����ƫ�� bias_updates += *(delta + i*outputs)
        axpy_cpu(l.outputs, 1, l.delta + i * l.outputs, 1, l.bias_updates, 1);
    }

    //BN �����   �ݶȱ�׼��
    if (l.batch_normalize) {
        // scale_updates += sigma(delta * x_norm)
        backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);

        // output *= scales
        scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);
        // delta ��ֵ
        mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
        // delta ����
        variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
        // delta ��̬��׼��
        normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float* a = l.delta;                 // �ݶ�
    float* b = state.input;             // ��һ��������Ϣ x
    float* c = l.weight_updates;        // c ���趨ΪȨ��  --> ����Ȩ�� (�� B �� C �����������)

    // Ȩ�ظ��� w += learn * gradient * x --> ���� l.delta Ϊ�ݶȵĺ���

    // ���±������ C += A*B*1
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;                    // c ���趨Ϊ ��һ���ݶ�  �����ǰһ���ݶ���Ϣ

    //����ǰһ���ݶ�
    if (c) gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);

    /// ����̫��
    /// 1. û�����֪ʶ����������ʶ�˽⣬�����ھ���
    /// 2. ˼ά��ʽ����(���δ֪�����ڼ����˽�������̬��-- > ģ��, ÿ���Լ�˼���Ľ�����һ�ֿ���, Ӧ�������˽�, ������Ϣ, ������˼��ÿһ���ĳ������)

    /// ����˼�� ��ͬ�������ѧϰ���䷶Χ�Ķ���
    /// 1. ����������ѧϰҪ��
    ///  1 > ����淶-- > ͳһ�淶������������Ч��
    ///  2 > ����ʵ�ַֿ顢ְ�ָܷ�˼��-- > ���㸴����ֲ�����Ч��
    ///  3 > ���������˼·-- > ��߿����ɹ���
    /// 2. Ƕ��ʽ��Ƭ���������ѧϰ�����䷶Χ�Ķ���
    ///  1 > ��Ƭ��ʹ�ã�Ƕ��ʽ��������-- > ����
    ///  2 > ����淶-- > ͳһ�淶������������Ч��
    ///  3 > ����ʵ�ַֿ顢ְ�ָܷ�˼��-- > ���㸴����ֲ�����Ч��
    ///  4 > ���������˼·-- > ��߿����ɹ���
    /// 3. �㷨���Կ���
    ///  1 > ���� ���Գ��򿪷�-- > ����
    ///  2 > ����淶-- > ͳһ�淶������������Ч��
    ///  3 > ����ʵ�ַֿ顢ְ�ָܷ�˼��-- > ���㸴����ֲ�����Ч��
    ///  4 > ���������˼·-- > ��߿����ɹ���
    /// 4. �㷨����
    ///  1 > �о����ֲ�ͬ���㷨ʵ��
    ///  2 > ������Ŀ����
    ///  3 > �����Ŀ������㷨ʵ��

    /// ѧ������̸����ѧϰ������ѧϰ����ĺ���̽��
    /// �����γ�һ���ض���ϰ��, �������µ�֪ʶ��
    /// 1. һ�״���淶(���ѧϰ) --> Ϊ����ҵ��Ա���ܺ�������߸������ŶӵĿ���Ч��(����)
    /// 2. һ���������˼·
    /// 3. һ������������
    /// 
    /// ��ҵ�ҵ�ѧϰ��������Ϊ�������ҵ����Ч��
    /// 
    /// ��֪��  �о���֪ʶ����Ķϲ� �� û�о��� �������ڵĴ���

}

//�Ǳ�׼�����Ӳ� 
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

//ͳ�� ȫ���Ӳ� ��Ϣ��ӡ
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
