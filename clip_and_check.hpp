#include<Eigen/Dense>

template <typename Derived>
void clip_and_check(Eigen::MatrixBase<Derived>& mat, const std::string& name, double clip_value = 1e6)
{
    mat = mat.cwiseMin(clip_value).cwiseMax(-clip_value);
    if (!mat.allFinite())
    {
        throw NumericalInstabilityError("Non-finite values detected in " + name);
    }
}

// Overload for Array types
template <typename Derived>
void clip_and_check(Eigen::ArrayBase<Derived>& arr, const std::string& name, double clip_value)
{
    arr = arr.min(clip_value).max(-clip_value);
    if (!arr.allFinite())
    {
        throw NumericalInstabilityError("Non-finite values detected in " + name);
    }
}