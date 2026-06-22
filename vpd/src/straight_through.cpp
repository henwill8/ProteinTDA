#include "straight_through.hpp"

// https://discuss.pytorch.org/t/torch-round-gradient/28628/9
torch::Tensor straight_through_round(torch::Tensor x) {
    return x + (torch::round(x) - x).detach();
}

class StraightThroughBincount : public torch::autograd::Function<StraightThroughBincount> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor indices, int64_t dim) {
            auto indices_int = torch::round(indices).to(torch::kInt64);
            ctx->save_for_backward({indices_int});
            return torch::bincount(indices_int, {}, dim).to(indices.options().dtype(torch::kFloat64));
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
            auto indices_int = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];
            auto grad_indices = torch::index_select(grad_output, 0, indices_int); // if loss increases for a bin, points in that bin get the gradient
            return {grad_indices, torch::Tensor()};
        }
};

torch::Tensor straight_through_bincount(torch::Tensor indices, int64_t dim) {
    return StraightThroughBincount::apply(indices, dim);
}
