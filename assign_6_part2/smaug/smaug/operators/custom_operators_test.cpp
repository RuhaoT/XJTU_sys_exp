#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/smaug_test.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/custom_operators.h"
#include "smaug/utility/debug_stream.h"
#include <iostream>

using namespace smaug;

// TEST_CASE_METHOD(SmaugTest, "MyCustomOperator", "[ops]") {
//   // DataLayout::NC is a simple 2D layout, where N = batches and C = a column
//   // of data.
//   std::vector<int> dims = {1, 10};
//   TensorShape shape(dims, DataLayout::NC);
//   Tensor* input0 = new Tensor("tensor0", shape);
//   // Allocate the memory for a 1x10 array of floats.
//   input0->allocateStorage<float>();
//   // Add some testing data.
//   input0->fillData<float>({1,2,3,4,5,6,7,8,9,10});
//   workspace()->addTensor(input0);

//   // Repeat this for a second input tensor.
//   Tensor* input1 = new Tensor("tensor1", shape);
//   input1->allocateStorage<float>();
//   input1->fillData<float>({2,3,4,5,6,7,8,9,10,11});
//   workspace()->addTensor(input1);
//   std::cout << "Prepared input tensors." << std::endl;

//   // Create the operator and fill it with our tensors.
//   using TestOp = MyCustomOperator<ReferenceBackend>;
//   auto op = new TestOp("eltwise_add", workspace());
//   op->setInput(input0, TestOp::kInput0);
//   op->setInput(input1, TestOp::kInput1);
//   op->createAllTensors();
//   // Allocates memory for all the output tensors created by createAllTensors.
//   allocateAllTensors<float>(op);
//   std::cout << "Created operator." << std::endl;

//   op->run();

//   std::cout << "Ran operator." << std::endl;
//   // Compare the output of the operator against expected values.
//   std::vector<float> expected_output = {3,5,7,9,11,13,15,17,19,21};
//   Tensor* output = op->getOutput(TestOp::kOutput);
//   // This performs an approximate comparison between the tensor's output and
//   // the expected values.
//   verifyOutputs(output, expected_output);
// }

// A function to fill the tensor with a sequence of monotonically increasing
// data, starting from 0. Note that this is ONLY advised for elementwise/unary
// operators in which we don't care about data in specific dimensions.
void fillTensorWithSequentialFloat32Data(Tensor* tensor) {
    float* data = tensor->data<float>();
    for (int i = 0; i < tensor->getShape().storageSize(); i++) {
        data[i] = i;
    }
}

void fillTensorWithSequentialFloat16Data(Tensor* tensor) {
    float16* data = tensor->data<float16>();
    for (int i = 0; i < tensor->getShape().storageSize(); i++) {
        data[i] = i;
    }
}

TEST_CASE_METHOD(SmaugTest, "MyCustomOperatorWithTiling", "[tiling]") {
    SECTION("SmvBackend") {

        // With float32 elements, this will occupy 128KB, which should create
        // four tiles per tensor.
        std::vector<int> dims = { 8, 128 };
        TensorShape shape(dims, DataLayout::NC);
        Tensor* input0 = new Tensor("tensor0", shape);
        Tensor* input1 = new Tensor("tensor1", shape);
        workspace()->addTensor(input0);
        workspace()->addTensor(input1);
        dout(1) << "Prepared input tensors.\n";

        // Create the operator and fill it with our tensors.
        using TestOp = MyCustomOperator<SmvBackend>;
        auto op = new TestOp("eltwise_add", workspace());
        op->setInput(input0, TestOp::kInput0);
        op->setInput(input1, TestOp::kInput1);
        op->createAllTensors();
        // This will handle creating/allocating storage/filling data into all
        // the input tensors.
        createAndFillTensorsWithData<float16>(
                op, &fillTensorWithSequentialFloat16Data);
        // Compute the expected output.
        std::vector<float16> expected_output(8 * 128, 0);
        for (int i = 0; i < expected_output.size(); i++) {
            expected_output[i] = 2 * i;  // 2 * 8 * 128 - 1 = 2047
        }
        dout(1) << "Expected output size: " << expected_output.size() << "\n";

        op->tile();

        dout(1) << "Tile operator.\n";

        op->run();

        Tensor* output = op->getOutput(TestOp::kOutput);
        verifyOutputs(output, expected_output);
    }
    // SECTION("ReferenceBackend") {
    //     // With float32 elements, this will occupy 128KB, which should create
    //     // four tiles per tensor.
    //     std::vector<int> dims = { 8, 4096 };
    //     TensorShape shape(dims, DataLayout::NC);
    //     Tensor* input0 = new Tensor("tensor0", shape);
    //     Tensor* input1 = new Tensor("tensor1", shape);
    //     workspace()->addTensor(input0);
    //     workspace()->addTensor(input1);
    //     dout(1) << "Prepared input tensors.\n";

    //     // Create the operator and fill it with our tensors.
    //     using TestOp = MyCustomOperator<ReferenceBackend>;
    //     auto op = new TestOp("eltwise_add", workspace());
    //     op->setInput(input0, TestOp::kInput0);
    //     op->setInput(input1, TestOp::kInput1);
    //     op->createAllTensors();
    //     // This will handle creating/allocating storage/filling data into all
    //     // the input tensors.
    //     createAndFillTensorsWithData<float>(
    //             op, &fillTensorWithSequentialFloat32Data);
    //     // Compute the expected output.
    //     std::vector<float> expected_output(8 * 128, 0);
    //     for (int i = 0; i < expected_output.size(); i++) {
    //         expected_output[i] = 2 * i;
    //     }
    //     dout(1) << "Expected output size: " << expected_output.size() <<
    //     "\n";

    //     op->tile();

    //     dout(1) << "Tile operator.\n";

    //     op->run();

    //     Tensor* output = op->getOutput(TestOp::kOutput);
    //     verifyOutputs(output, expected_output);
    // }
}