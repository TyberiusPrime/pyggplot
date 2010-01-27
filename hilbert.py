import math
import time
import numpy
import scipy.weave
import unittest


def hilbert_plot(vector, bitmap_width):
    """Take a numpy 1d array and plot it into a 2d array of bitmap_width * bitmap_width,
    for example for plotting"""
    bitmap = numpy.zeros((bitmap_width, bitmap_width,), dtype=vector.dtype)
    #reps = int(math.ceil(math.log(vector.shape[0], 4)))
    reps = int(math.ceil(math.log(bitmap_width * bitmap_width, 4)))
    hilbert_coordinates = hilbert_weave(reps)
    draw_coordinates = numpy.array(hilbert_coordinates * bitmap_width, dtype=numpy.uint32)

    draw_data = numpy.interp(numpy.array(range(0, bitmap_width * bitmap_width)) * 
                             vector.shape[0] / float(bitmap_width * bitmap_width),
                             range(0, vector.shape[0]),
                             vector)
    for value, (x,y) in zip(draw_data, draw_coordinates):
        bitmap[x,y] = value
    return bitmap


  
def hilbert_python(points_on_curve):
    """calculate hilbert curve points in python"""
    pos = [0]
    def inner_hilbert(x0, y0, xi, xj, yi, yj, n, col):
        if n <= 0:
            X = x0 + (xi + yi)/2
            Y = y0 + (xj + yj)/2
            
            # Output the coordinates of the cv
            #print '%s %s 0' % (X, Y)
            col[pos[0]] = (X, Y)
            pos[0] += 1
        else:
            inner_hilbert(x0,               y0,               yi/2, yj/2, xi/2, xj/2, n - 1, col)
            inner_hilbert(x0 + xi/2,        y0 + xj/2,        xi/2, xj/2, yi/2, yj/2, n - 1, col)
            inner_hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, n - 1, col)
            inner_hilbert(x0 + xi/2 + yi,   y0 + xj/2 + yj,  -yi/2,-yj/2,-xi/2,-xj/2, n - 1, col)
    collector = numpy.zeros((4**(points_on_curve), 2), dtype=numpy.float)
    inner_hilbert(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, points_on_curve, collector)
    return collector



def hilbert_weave(reps):
    """calculate hilbert curve points in c - much faster"""
    reps = int(reps)
    collector = numpy.zeros((4**(reps), 2), dtype=numpy.double)
    scipy.weave.inline("""
int pos = 0;
hilbert(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, reps, collector, &pos);
                      """, 
                       {
                           'reps': reps,
                           'collector': collector,
                       },
                      support_code = """
void hilbert(double x0, double y0, double xi, double xj, double yi, double yj, int n, double* col, int* pos)
{
    if (n <= 0)
    {
         double X = x0 + (xi + yi)/2;
         double Y = y0 + (xj + yj)/2;
        int ii = *pos;
        col[ii * 2] = X;
        col[ii * 2 + 1] = Y;
        *pos += 1;
    }
    else
    {
        hilbert(x0,               y0,               yi/2, yj/2, xi/2, xj/2, n - 1, col, pos);
        hilbert(x0 + xi/2,        y0 + xj/2,        xi/2, xj/2, yi/2, yj/2, n - 1, col, pos);
        hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, n - 1, col, pos);
        hilbert(x0 + xi/2 + yi,   y0 + xj/2 + yj,  -yi/2,-yj/2,-xi/2,-xj/2, n - 1, col, pos);
    }
}
"""
                      )
    return collector
        
class HilbertTests(unittest.TestCase):

    def test_python_simple(self):
        col = hilbert_python(1)
        self.assertEqual(len(col), 4)
        self.assertAlmostEqual(col[0,0], 0.25)
        self.assertAlmostEqual(col[0,1], 0.25)
        self.assertAlmostEqual(col[1,0], 0.75)
        self.assertAlmostEqual(col[1,1], 0.25)
        self.assertAlmostEqual(col[2,0], 0.75)
        self.assertAlmostEqual(col[2,1], 0.75)
        self.assertAlmostEqual(col[3,0], 0.25)
        self.assertAlmostEqual(col[3,1], 0.75)

    def test_c_against_python(self):
        for reps in xrange(0, 9):
            start = time.time()
            col = hilbert_python(reps)
            col2 = hilbert_weave(reps)
            self.assertFalse((col != col2).any())

    def test_plot(self):
        vector = numpy.array([1,2,3,4], dtype=numpy.int8)
        plot = hilbert_plot(vector, 16)
        print plot
        self.assertEqual(plot.shape[0], 16)
        self.assertEqual(plot.shape[1], 16)
        self.assertEqual(plot[4,4], 1)
        self.assertEqual(plot[12,4], 2)
        self.assertEqual(plot[12,12], 3)
        self.assertEqual(plot[4,12], 4)

    def test_plot2(self):
        vector = numpy.array([1,2,3,55], dtype=numpy.int8)
        plot = hilbert_plot(vector, 2)
        self.assertEqual(plot.shape[0], 2)
        self.assertEqual(plot.shape[1], 2)
        self.assertEqual(plot[0,0], 1)
        self.assertEqual(plot[1,0], 2)
        self.assertEqual(plot[1,1], 3)
        self.assertEqual(plot[0,1], 55)

    def test_plot3(self):
        vector = numpy.array([1,2,3,0], dtype=numpy.int8)
        plot = hilbert_plot(vector, 2)
        self.assertEqual(plot.shape[0], 2)
        self.assertEqual(plot.shape[1], 2)
        self.assertEqual(plot[0,0], 1)
        self.assertEqual(plot[1,0], 2)
        self.assertEqual(plot[1,1], 3)
        self.assertEqual(plot[0,1],0)

    def test_plot4(self):
        vector = numpy.array(range(0, 8 ), dtype=numpy.double)
        plot = hilbert_plot(vector, 2)
        self.assertEqual(plot.shape[0], 2)
        self.assertEqual(plot.shape[1], 2)
        print plot

if __name__ == '__main__': 
    unittest.main()
