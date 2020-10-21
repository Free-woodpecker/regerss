#ifndef __Kalman_h_
#define __Kalman_h_


class Kalman {
public:
	Kalman(float Q, float R) {
		_q = Q;
		_r = R;

		Reset();
	}

	void Reset() {
		_xPosteriEstimate = 0;
		_errPosteriEstimate = 0;
		_xPrioriEstimate = 0;
		_errPrioriEstimate = 0;
		
		_a = 1;
		_h = 1;
		_k = 0;
	}

	float Update(float in) {
		_xPrioriEstimate = _a * _xPosteriEstimate;									//更新x的先验估计
		_errPrioriEstimate = _a * _errPosteriEstimate + _q;							//更新误差的先验估计

		_k = _errPrioriEstimate / (_errPrioriEstimate + _r);						//更新增益系数
		_xPosteriEstimate = _xPrioriEstimate + _k * (in - _h * _xPrioriEstimate);	//更新x的后验估计
		_errPosteriEstimate = (1 - _k * _h) * _errPrioriEstimate;					//更新误差的后验估计
		return _xPosteriEstimate;
	}

private:
	float _q;                   // 过程方差
	float _r;                   // 测量方差估计

	float _a;
	float _h;

	float _xPosteriEstimate;    // 
	float _errPosteriEstimate;	// 
	float _xPrioriEstimate;		// 
	float _errPrioriEstimate;	// 
	float _k;					// 增益系数
};




#endif

