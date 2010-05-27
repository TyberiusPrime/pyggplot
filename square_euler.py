import cairo

class Square_Euler:
    """Plot a square pseudo euler diagram of two groups of sets,
    where within a group the sets are non-overlapping"""

    def __init__(self, class_tuples, fill_percentages = None):
        self.class_tuples = class_tuples
        self.fill_colors = [
            (0.0, 0, 1, 0.5),
            (1.0, 0, 0, 0.5),
            (0.0, 0.8, 0, 0.5),
            (0.0, 0, 0, 0.5),
            (0.3, 0, 0.3, 0.5),
        ]
        self.fill_percentages = fill_percentages
        if len(self.class_tuples) != 4:
            raise ValueError("Square_Euler takes exactly four tuples of (name, set)")
        try:
            self.assert_distinct(class_tuples[0][1], class_tuples[2][1])
        except ValueError:
            raise ValueError("Non-overlapping sets were not distinct! %s %s"%  (class_tuples[0][0], class_tuples[2][0]))
        try:
            self.assert_distinct(class_tuples[1][1], class_tuples[3][1])
        except ValueError:
            raise ValueError("Non-overlapping sets were not distinct! %s %s"%  (class_tuples[1][0], class_tuples[3][0]))

    def assert_distinct(self, setA, setB):
        if len(setA.intersection(setB)) > 0:
            raise ValueError("Non-overlapping sets were not distinct!")


    def plot(self, output_filename, width = 10, height=10, dpi=300):
        image_width = width * dpi
        image_height = height * dpi
        if output_filename.endswith('.pdf'):
            surface = cairo.PDFSurface(output_filename, image_width, image_height) 
        elif output_filename.endswith('.png'):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,image_width, image_height)

        rect_size = 3.0 / 7.0

        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1,1,1)
        ctx.rectangle(0,0,image_width, image_height)
        ctx.fill()
        ctx.set_source_rgb(0,0,0)



        if self.fill_percentages:
            ctx.set_source_rgba(*self.fill_colors[0])
            ctx.rectangle(  image_width / 2.0 - (image_width * rect_size / 2.0),
                        image_height * rect_size,
                        (image_width * rect_size),
                        (image_height * rect_size) * self.fill_percentages[0] * -1)
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[1])
            ctx.rectangle(   image_width - (image_width * rect_size),
                        (image_height / 2.0) + (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        (image_height * rect_size) * self.fill_percentages[2] * -1 )

            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[2])
            ctx.rectangle(
                        image_width / 2.0 - (image_width * rect_size / 2.0),
                        image_height,
                        (image_width * rect_size),
                        -1 * (image_height * rect_size) * self.fill_percentages[4])
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[3])
            ctx.rectangle(   0,
                        (image_height / 2.0) + (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        -1 * (image_height * rect_size)  * self.fill_percentages[6] )

            ctx.fill()
            ctx.set_source_rgba(1,1,1,1)
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0)
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0)
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height - image_height * rect_size,
                        image_width * rect_size / 3.0,
                        image_height * rect_size / 3.0)
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height - image_height * rect_size,
                        image_width * rect_size / 3.0,
                        image_height * rect_size / 3.0)
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[4])
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[7])
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[1])
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height - image_height * rect_size + image_height * rect_size / 3.0,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[3])
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height - image_height * rect_size + image_height * rect_size / 3.0,
                        image_width * rect_size / 3.0,
                        -1 *  image_height * rect_size / 3.0 * self.fill_percentages[5])
            ctx.fill()
        ctx.set_source_rgba(0,0,0,1)

        ctx.rectangle(  image_width / 2.0 - (image_width * rect_size / 2.0),
                        0,
                        (image_width * rect_size),
                        (image_height * rect_size))
        ctx.rectangle(   image_width - (image_width * rect_size),
                        (image_height / 2.0) - (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        (image_height * rect_size) )
        ctx.rectangle(
                        image_width / 2.0 - (image_width * rect_size / 2.0),
                        image_height - (image_height * rect_size),
                        (image_width * rect_size),
                        (image_height * rect_size))
        ctx.rectangle(   0,
                        (image_height / 2.0) - (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        (image_height * rect_size) )
        ctx.stroke()

        ctx.set_font_size(image_height / 20)

        def rendered_text(x, y, text):
            lines = text.split("\n")
            if len(lines) == 1:
                ext = ctx.text_extents(lines[0])
                ctx.move_to(x - ext[2] / 2.0, y + ext[3] / 2.0)
                ctx.show_text(lines[0])
            else:
                widths = []
                for line in lines:
                    ext = ctx.text_extents(line)
                    widths.append(ext[2])
                line_height = ext[3] * 1.05
                for ii, line in enumerate(lines):
                    ctx.move_to(x - widths[ii] / 2.0, y + line_height * ii)
                    ctx.show_text(line)

        rendered_text( image_width / 2.0, image_height * rect_size / 2.0, 
                      "%s\n%i" % (self.class_tuples[0][0],
                                 len(self.class_tuples[0][1] - self.class_tuples[1][1] - self.class_tuples[3][1])
                                 )
                     )
        rendered_text( image_width - image_width * rect_size / 2.0, image_height / 2.0, 
                      "%s\n%i" % (self.class_tuples[1][0],
                                 len(self.class_tuples[1][1] - self.class_tuples[0][1] - self.class_tuples[2][1])
                                 )
                     )
        rendered_text( image_width / 2.0, image_height - image_height * rect_size / 2.0, 
                      "%s\n%i" % (self.class_tuples[2][0],
                                 len(self.class_tuples[2][1] - self.class_tuples[1][1] - self.class_tuples[3][1])
                                 )
                     )
        rendered_text( image_width * rect_size / 2.0, image_height / 2.0, 
                      "%s\n%i" % (self.class_tuples[3][0],
                                 len(self.class_tuples[3][1] - self.class_tuples[0][1] - self.class_tuples[2][1])
                                 )
                     )

        rendered_text((image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (len(self.class_tuples[0][1].intersection(self.class_tuples[3][1])),))

        rendered_text(image_width - (image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (len(self.class_tuples[1][1].intersection(self.class_tuples[0][1])),))

        rendered_text(image_width - (image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      image_height - (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (len(self.class_tuples[2][1].intersection(self.class_tuples[1][1])),))

        rendered_text((image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      image_height - (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (len(self.class_tuples[3][1].intersection(self.class_tuples[2][1])),))



            
        if output_filename.endswith('.pdf'):
            pass #taken care of when initializing the surface
        else:
            surface.write_to_png(output_filename)
        surface.finish()

if __name__ == '__main__': 
    x = Square_Euler([
        ('shu', set([1,2,3])),
        ('sha', set([5,4,3])),
        ('sho', set([6,55,7])),
        ('she', set([1,2,7])),
    ],
        [0.5, 0.3, 0.8, 0.88, 0.75, 0.2, 0.1, 0.55]
    
    )
    x.plot("test.png")
