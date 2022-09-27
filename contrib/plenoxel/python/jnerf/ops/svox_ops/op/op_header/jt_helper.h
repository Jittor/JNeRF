#pragma once
#include "var.h"
#define HOST_DEVICE __host__ __device__
typedef jittor::Var Var;
template <typename T, size_t D, typename index_t = int32_t>
class PackedVarBase {
   public:
    T* _data;
    size_t _dim;
    index_t _shape[D];
    index_t _stride[D];
    index_t _num;

    HOST_DEVICE PackedVarBase(T* data, const index_t* shapes, const index_t* strides) {
        _data = data;
        _dim = D;
        for (int i = 0; i < D; i++) {
            _shape[i] = shapes[i];
            _stride[i] = strides[i];
        }
    }
    __host__ PackedVarBase(Var* var) {
        _data = var->ptr<T>();
        _dim = D;
        _num = 1;
        for (int i = 0; i < D; i++) {
            _shape[i] = var->shape[i];
            _num = _num * _shape[i];
        }
        _stride[D - 1] = 1;
        for (int i = D - 2; i >= 0; i--) {
            _stride[i] = _shape[i + 1] * _stride[i + 1];
        }
    }
    template <typename source_index_t>
    HOST_DEVICE PackedVarBase(T* data, const source_index_t* shapes, const source_index_t* strides) {
        _data = data;
        _dim = D;
        for (int i = 0; i < D; i++) {
            this->_shape[i] = shapes[i];
           this-> _stride[i] = strides[i];
        }
    }


    HOST_DEVICE T* data() {
        return _data;
    }
    HOST_DEVICE index_t dim() {
        return _dim;
    }
    HOST_DEVICE index_t stride(index_t i) const {
        return _stride[i];
    }
    HOST_DEVICE const T* data() const {
        return _data;
    }
    HOST_DEVICE const index_t size(index_t i) const {
        return _shape[i];
    }
};
template <typename T, size_t D, typename index_t = int32_t>
class PackedVar : public PackedVarBase<T, D, index_t> {
   public:
    HOST_DEVICE PackedVar(T* data, const index_t* shapes, const index_t* strides)
        : PackedVarBase<T, D, index_t>(data, shapes, strides) {
        // printf("1111\n");
    }
    __host__ PackedVar(Var* var)
        : PackedVarBase<T, D, index_t>(var) {
        //  printf("222\n");
    }
    template<typename source_index_t>
    HOST_DEVICE PackedVar(T* data, const source_index_t* shapes, const source_index_t* strides)
        : PackedVarBase<T, D, index_t>(data, shapes, strides) {
        // printf("1111\n");
    }
    // T* _data;
    // size_t _dim;
    // int _shape[D];
    // int _stride[D];
    // int _num;
    // HOST_DEVICE PackedVar(T* data) {
    //     _data = data;
    //     _dim = D;
    //     // printf("packedvar11111111\n");
    //     // for (int i = 0; i < D; i++) {
    //     //     _shape = shapes[i];
    //     //     _stride = strides[i];
    //     // }
    // }
    // HOST_DEVICE PackedVar(T* data, int* shapes, int* strides) {
    //     _data = data;
    //     _dim = D;
    //     for (int i = 0; i < D; i++) {
    //         _shape[i] = shapes[i];
    //         _stride[i] = strides[i];
    //     }
    // }
    // __host__ PackedVar(Var* var) {
    //     _data = var->ptr<T>();
    //     _dim = D;
    //     _num = 1;
    //     for (int i = 0; i < D; i++) {
    //         _shape[i] = var->shape[i];
    //         _num = _num * _shape[i];
    //     }
    //     _stride[D - 1] = 1;
    //     for (int i = D - 2; i >= 0; i--) {
    //         _stride[i] = _shape[i + 1] * _stride[i + 1];
    //     }
    // }

    // HOST_DEVICE T* data() {
    //     return _data;
    // }
    // HOST_DEVICE int dim() {
    //     return _dim;
    // }

    // HOST_DEVICE const T* data() const {
    //     return _data;
    // }
    // HOST_DEVICE const size_t size(size_t i) const {
    //     return _shape[i];
    // }
    HOST_DEVICE PackedVar<T, D - 1,index_t> operator[](index_t i) {
        index_t* new_shape = this->_shape + 1;
        index_t* new_stride = this->_stride + 1;
        return PackedVar<T, D - 1, index_t>(this->_data + this->_stride[0] * i, new_shape, new_stride);
    }
    HOST_DEVICE const PackedVar<T, D - 1,index_t> operator[](index_t i)const {
        // index_t* new_shape = this->_shape + 1;
        // index_t* new_stride = this->_stride + 1;
        return PackedVar<T, D - 1, index_t>(this->_data + this->_stride[0] * i, this->_shape + 1, this->_stride + 1);
    }
    HOST_DEVICE T& operator()(size_t i) {
        return this->_data[i];
    }
      HOST_DEVICE const T& operator()(size_t i)const {
        return this->_data[i];
    }
};
template <typename T, typename index_t>
class PackedVar<T, 1, index_t> : public PackedVarBase<T, 1, index_t> {
   public:
    HOST_DEVICE PackedVar(T* data, const index_t* shapes, const index_t* strides)
        : PackedVarBase<T, 1, index_t>(data, shapes, strides) {
        // printf("3333\n");
    }
    __host__ PackedVar(Var* var)
        : PackedVarBase<T, 1, index_t>(var) {
        // printf("444\n");
    }
    template<typename source_index_t>
     HOST_DEVICE PackedVar(T* data, const source_index_t* shapes, const source_index_t* strides)
        : PackedVarBase<T, 1, index_t>(data, shapes, strides) {
        // printf("3333\n");
    }
    HOST_DEVICE T& operator[](index_t i) {
        return this->_data[this->_stride[0] * i];
    }
     HOST_DEVICE const T& operator[](index_t i) const{
        return this->_data[this->_stride[0] * i];
    }
    
};

// template <typename T>
// T& PackedVar<T, 1>::operator[](int i) {
//     return this->_data[this->_stride[0]*i];
// }
template <typename T, size_t D>
using PackedVar32 = PackedVar<T, D, int32_t>;
template <typename T, size_t D>
using PackedVar64 = PackedVar<T, D, int64_t>;
