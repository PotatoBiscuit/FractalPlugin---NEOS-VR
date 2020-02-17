using FrooxEngine.LogiX;
using FrooxEngine;
using BaseX;

namespace FractalLibrary
{
    [Category("LogiX/Fractals")]
    class JuliaInput : LogixNode
    {
        public readonly Input<double2> customCValue;
        public readonly Output<FractalData> fractalData;

        protected override void OnStart()
        {
            base.OnStart();
            fractalData.Value = new FractalData();
            ChangeFractalData();
        }

        protected override void OnInputChange()
        {
            base.OnInputChange();
            ChangeFractalData();
        }

        private void ChangeFractalData()
        {
            if ( customCValue.IsConnected )
            {
                fractalData.Value.customCorZValue = customCValue.Evaluate();
            }
            else
            {
                fractalData.Value.customCorZValue = new double2(-.8f, .156f);
            }
            fractalData.Value.fractalType = FractalType.julia;
        }
    }
}
