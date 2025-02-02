#ifndef UTILS_H_
#define UTILS_H_

#ifndef MSE_LOSS
#define MSE_LOSS(gt, pred) mse_loss(gt, pred) 
#endif // MSE_LOSS

// This header contains definitions of loss functions

// Spirit: Matrix Inputs have Output in MAtrix form!


float mse_loss(Mat gt, Mat pred);

#endif // UTILS_H_

#ifdef UTILS_IMPLEMENTATION
float mse_loss(Mat gt, Mat pred) {
    float loss = 0.f;
    int n = (gt.rows)*(gt.cols);
    for (size_t i = 0; i < gt.rows; ++i) {
        for (size_t j = 0; j < gt.cols; ++j) {
            loss += pow((MAT_AT(gt, i, j) - MAT_AT(pred, i, j)), 2);
        }
    }
    return loss / n;
}



#endif // UTILS_H_IMPLEMENTATION
