#include "smaug/operators/custom_operators.h"
#include "fp16.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/utility/debug_stream.h"
#include "smaug/core/backend.h"
#include <array>
#include <iostream>

#define __TEST__

namespace smaug {

template <typename Backend>
void MyCustomOperator<Backend>::createAllTensors() {
    Tensor* output = new Tensor(name, inputs.at(kInput0)->getShape());
    outputs.at(kOutput) = output;
    workspace->addTensor(output);
}

#ifdef __TEST__
template <typename Backend>
void MyCustomOperator<Backend>::elementwise_add_float32(float* input0,
                                                        float* input1,
                                                        float* output,
                                                        int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input0[i] + input1[i];
        dout(2) << "adding " << input0[i] << " and " << input1[i] << " to get "
                << output[i] << "\n";
    }
}

template <typename Backend>
void MyCustomOperator<Backend>::elementwise_add_float16(float16* input0,
                                                        float16* input1,
                                                        float16* output,
                                                        int size) {
    for (int i = 0; i < size; i++) {
        // float16 = unsigned short int
        // CANNOT directly add float16 values
        // output[i] = input0[i] + input1[i]; <-- ERROR
        // use fp16() and fp32() to convert between float16 and float32
        output[i] =
                fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(input0[i]) +
                                          fp16_ieee_to_fp32_value(input1[i]));
        dout(2) << "adding " << input0[i] << " and " << input1[i] << " to get "
                << output[i] << "\n";
    }
}
#else
// By convention, we prefix all pointers into host memory with "host_".
template <typename Backend>
void MyCustomOperator<Backend>::elementwise_add_ref(float* host_input0,
                                                           float* host_input1,
                                                           float* host_output,
                                                           float* spad0,
                                                           float* spad1,
                                                           int size) {
    // Copy input data from host_inputN to spadN. The first argument to
    // dmaLoad or dmaStore is always the destination.
    dout(1) << "Copying input data to scratchpads\n";
    dmaLoad(host_input0, host_input0, size * sizeof(float));
    dmaLoad(host_input1, host_input1, size * sizeof(float));
    dout(1) << "Running elementwise_add_ref\n";
    dout(2) << "size: " << size << "\n";
    for (int i = 0; i < size; i++) {
        // Accumulate the data from spad0 into spad1.
        // NOTE: This could be optimized more if we had three scratchpads
        // instead of two. This would be a great exercise for the reader :)
        dout(2) << "adding " << spad0[i] << " and " << spad1[i] << " to get "
                << spad0[i] + spad1[i] << "\n";
        host_output[i] = host_input0[i] + host_input1[i];
    }
    // Copy output data from spad1 back to the host.
    dout(1) << "Copying results back to host\n";
    dmaStore(host_output, host_output, size * sizeof(float));
}
#endif

template <typename Backend>
void MyCustomOperator<Backend>::run() {
    TiledTensor& input0 = tiledTensors[kInput0];
    TiledTensor& input1 = tiledTensors[kInput1];
    TiledTensor& output = tiledTensors[kOutput];
    Tensor* outputTensor = getOutput(kOutput);

    dout(1) << "Running MyCustomOperator\n";

    for (int i = 0; i < input0.size(); i++) {
        Tensor* input0Tile = input0.getTileWithData(i);
        Tensor* input1Tile = input1.getTileWithData(i);
        Tensor* outputTile = output.getTileWithData(i);

        // Get handles to the actual underlying data storage. This performs
        // a dynamic_cast to the specified data type, which we verified is
        // safe inside validate().

        // obtain backend data type
        if (Backend::Alignment == 0) {
            dout(1) << "Backend is ReferenceBackend" << "\n";
            dout(2) << "Datatype: " << input0Tile->getDataType() << "\n";
            float* input0Data = input0Tile->data<float>();
            float* input1Data = input1Tile->data<float>();
            float* outputData = outputTile->data<float>();

#ifdef __TEST__

            elementwise_add_float32(input0Data,
                                    input1Data,
                                    outputData,
                                    outputTile->getShape().size());
#else
            int size = input0Tile->getShape().storageSize() * sizeof(float);
            dout(1) << "Mapping arrays to accelerator\n";
            // Set up the TLB mappings.
            mapArrayToAccelerator(
                    ref::kMyCustomOperatorHw,
                    "host_input0", 
                    input0Data,
                    size            
            );
            mapArrayToAccelerator(ref::kMyCustomOperatorHw, "host_input1",
                                  input1Data, size);
            mapArrayToAccelerator(ref::kMyCustomOperatorHw, "host_output",
                                  outputData, size);
            dout(1) << "Invoking kernel\n";

            // Wrap the call to elementwise_add_ref with invokeKernel.
            invokeKernel(ref::kMyCustomOperatorHw,  // our accelerator ID
                         elementwise_add_ref,  // if not simulating, the
                                               // function to call
                         // All of the function call arguments.
                         input0Data,
                         input1Data,
                         outputData,
                         ref::spad0,
                         ref::spad1,
                         input0Tile->getShape().storageSize());
#endif

        } else {
            dout(1) << "Backend is SmvBackend\n";
            dout(2) << "Datatype: " << input0Tile->getDataType() << "\n";
            float16* input0Data = input0Tile->data<float16>();
            float16* input1Data = input1Tile->data<float16>();
            float16* outputData = outputTile->data<float16>();

#ifdef __TEST__

            elementwise_add_float16(input0Data,
                                    input1Data,
                                    outputData,
                                    outputTile->getShape().size());
#else
            int size = input0Tile->getShape().storageSize() * sizeof(float16);

            dout(1) << "Mapping arrays to accelerator\n";
            // Set up the TLB mappings.
            mapArrayToAccelerator(
                    smv::kMyCustomOperatorHw,  // The accelerator ID this
                                                   // TLB mapping is for.
                    "host_input0",  // The name of the function argument in
                                    // the kernel function.
                    input0Data,     // The pointer to the data.
                    size            // The size of the TLB mapping
            );
            mapArrayToAccelerator(smv::kMyCustomOperatorHw, "host_inputs1",
                                  input1Data, size);
            mapArrayToAccelerator(smv::kMyCustomOperatorHw, "host_results",
                                  outputData, size);

            dout(1) << "Invoking kernel\n";
            // Wrap the call to elementwise_add_ref with invokeKernel.
            invokeKernel(smv::kMyCustomOperatorHw,  // our accelerator ID
                         smv_eltwise_add_nc_vec_fxp,  // if not simulating, the
                                                      // function to call
                         // All of the function call arguments.
                         input0Data,
                         input1Data,
                         outputData,
                         smv::spad0,
                         smv::spad1,
                         smv::spad2,
                         outputTile->getShape().storageSize());
#endif
        }

        // std::cout << "Finished iteration " << i << std::endl;
    }
    // The results of the elementwise_add_ref are stored in the tiled tensor. We
    // need to merge the data from the individual tiles back into a single
    // contiguous Tensor.
    flattenTiledTensor(tiledTensors[kOutput], outputTensor);
    // print the output tensor
    dout(3) << "Output tensor:\n";
    if (outputTensor->getDataType() == DataType::Float16) {
        float16* outputData = outputTensor->data<float16>();
        for (int i = 0; i < outputTensor->getShape().storageSize(); i++) {
            dout(3) << outputData[i] << " ";
        }
    } else {
        float* outputData = outputTensor->data<float>();
        for (int i = 0; i < outputTensor->getShape().storageSize(); i++) {
            dout(3) << outputData[i] << " ";
        }
        dout(3) << "\n";
    }
}

template <typename Backend>
bool MyCustomOperator<Backend>::validate() {

    Tensor* input0 = getInput(kInput0);
    Tensor* input1 = getInput(kInput1);
    // support the default data types for both backends.

    return (input0->getShape() == input1->getShape() ||
            (input0->getDataType() != DataType::Float32 &&
             input0->getDataType() != DataType::Float16) ||
            (input1->getDataType() != DataType::Float32 &&
             input1->getDataType() != DataType::Float16));
};

template <typename Backend>
void MyCustomOperator<Backend>::tile() {
    auto inputs0 = getInput(kInput0);
    auto inputs1 = getInput(kInput1);
    auto outputs = getOutput(kOutput);
    dout(1) << "Tiling: Fetched input tensors.\n";
    // The simplest tiling strategy is to tile per batch. Each tile will
    // have a size of at most 1 x maxTileSize.
    int maxTileSize = std::min(Backend::SpadSize() / inputs0->getDataTypeSize(),
                               inputs0->getShape().storageSize());
    maxTileSize = std::max(maxTileSize, 1);
    dout(2) << "inputs0 size: " << inputs0->getShape().storageSize();
    dout(2) << "Max tile size: " << maxTileSize;
    TensorShape tileShape(
            { 1, maxTileSize }, DataLayout::NC, Backend::Alignment);
    // The final bool parameter specifies whether to copy the data from
    // the source tensor into each of its tiles. Obivously, we want to
    // do this for the input tensors, but the output tensor is empty, so
    // there's no need to waste time on that.
    dout(1) << "Tiling: Generating tiled tensors.\n";
    tiledTensors[0] =
            generateTiledTensorPerBatchNC(inputs0, tileShape, this, true);
    tiledTensors[1] =
            generateTiledTensorPerBatchNC(inputs1, tileShape, this, true);
    dout(1) << "Tiling: Finished generating input tensors.\n";
    tiledTensors[2] =
            generateTiledTensorPerBatchNC(outputs, tileShape, this, false);
    dout(1) << "Tiling: Finished generating tiled tensors.\n";
};
}  // namespace smaug