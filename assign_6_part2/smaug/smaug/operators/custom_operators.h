#include "fp16.h"
#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"
#include "smaug/utility/debug_stream.h"
#include <array>
#include <iostream>

namespace smaug {

template <typename Backend>
class MyCustomOperator : public Operator {
   public:
    MyCustomOperator(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::CustomOp, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void setParam1(int val) { param1 = val; }
    void setParam2(int val) { param2 = val; }

    enum { kInput0, kInput1, kNumInputs };
    enum { kOutput, kNumOutputs };

    void createAllTensors() override;

    void elementwise_add_float32(float* input0,
                                 float* input1,
                                 float* output,
                                 int size);
    void elementwise_add_float16(float16* input0,
                                 float16* input1,
                                 float16* output,
                                 int size);
    static void elementwise_add_ref(float* host_input0,
                                    float* host_input1,
                                    float* host_output,
                                    float* spad0,
                                    float* spad1,
                                    int size);
    

    // A required function that implements the actual Operator logic.  Leave
    // this blank for now.
    void run() override;


    // Optional but recommended function to verify operator parameters.
    bool validate() override;

    // An optional function to tile the input tensors.
    void tile() override;


   private:
    int param1 = 0;
    int param2 = 0;
    // Because tensor tiling is done at the start of the program (before the
    // operator starts running), these tiles need to be stored in memory for
    // use later.
    std::array<TiledTensor, 3> tiledTensors;
};

//explicit instantiation for compiling
template class MyCustomOperator<ReferenceBackend>;
template class MyCustomOperator<SmvBackend>;

}  // namespace smaug