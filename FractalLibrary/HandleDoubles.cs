using FrooxEngine.LogiX;
using FrooxEngine;
using BaseX;

namespace FractalLibrary
{
    [Category("LogiX/Input")]
    class FloatToDouble : LogixNode
    {
        public readonly Input<float> input;
        public readonly Output<double> output;

        protected override void OnInputChange()
        {
            output.Value = input.Evaluate(0);
        }
    }

    [Category("LogiX/Input")]
    class PackDoubleXY : LogixNode
    {
        public readonly Input<double> input1;
        public readonly Input<double> input2;

        public readonly Output<double2> packedOutput;

        protected override void OnInputChange()
        {
            packedOutput.Value = new double2(input1.Evaluate(0), input2.Evaluate(0));
        }
    }
}
