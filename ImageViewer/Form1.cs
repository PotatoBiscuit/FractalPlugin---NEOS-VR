using System.Drawing;
using System.Windows.Forms;

namespace ImageViewer
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            PictureBox viewingImage = new PictureBox();
            FractalViewer fractViewer = new FractalViewer();
            viewingImage.Image = fractViewer.CreateNewton();
            viewingImage.Height = viewingImage.Image.Height;
            viewingImage.Width = viewingImage.Image.Width;
            viewingImage.Location = new Point(0, 0);

            Height = viewingImage.Height;
            Width = viewingImage.Width;
            Controls.Add(viewingImage);
        }
    }
}
