import cairo
import exptools

class SquareEulerFromNumbers:

    def __init__(self, class_names, counts, fill_percentages = None):
        """All values are in clockwise order. So first A, then A and B, then B, then B and C, then C...
        """
        self.class_names = class_names
        self.counts = counts
        if fill_percentages and len(fill_percentages) != 8:
            raise ValueError("Fill percentages takes exactly 8 values")
        self.fill_colors = [
                (0, 100.0 / 255, 0, 1), #darkgreen
                (0.3,.3,.3,1), #grey
                (1,0,0,1), #red
                (0.3,.3,.3,1), #grey
                (0,0,1,1), # blue,
                (0.3,.3,.3,1), #grey
                (160.0 / 255, 32.0 / 255, 240.0 / 255, 1), # purple,
                (0.3,.3,.3,1), #grey
        ]
        self.background_color = (1,1,1,1)
        self.border_color = (0,0,0,1) # black,
        self.text_colors = [
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,
                (0,0,0,1), # black,

        ]
        self.fill_percentages = fill_percentages
        if len(self.class_names) != 4:
            raise ValueError("Input error. 4 Classes, 8 sets of counts")
   

    def plot(self, output_filename, width = 10, height=10, dpi=300):
        if len(self.text_colors) != 8:
            raise ValueError("Need exactly 8 text colors")
        if len(self.fill_colors) != 8:
            raise ValueError("Need exactly 8 fill colors")
        image_width = width * dpi
        image_height = height * dpi
        if output_filename.endswith('.pdf'):
            surface = cairo.PDFSurface(output_filename, image_width, image_height) 
        elif output_filename.endswith('.png'):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,image_width, image_height)
	else:
            raise ValueError("Invalid output file for plot %s" % output_filename)

        rect_size = 3.0 / 7.0

        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background_color)
        ctx.rectangle(0,0,image_width, image_height)
        ctx.fill()
        ctx.scale(0.95, 0.95)
        ctx.translate(image_width * 0.025, image_height * 0.025)

        if self.fill_percentages:
            #fill big rectangles
            ctx.set_source_rgba(*self.fill_colors[0])
            ctx.rectangle(  image_width / 2.0 - (image_width * rect_size / 2.0),
                        image_height * rect_size,
                        (image_width * rect_size),
                        (image_height * rect_size) * self.fill_percentages[0] * -1)
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[2])
            ctx.rectangle(   image_width - (image_width * rect_size),
                        (image_height / 2.0) + (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        (image_height * rect_size) * self.fill_percentages[2] * -1 )
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[4])
            ctx.rectangle(
                        image_width / 2.0 - (image_width * rect_size / 2.0),
                        image_height,
                        (image_width * rect_size),
                        -1 * (image_height * rect_size) * self.fill_percentages[4])
            ctx.fill()
            ctx.set_source_rgba(*self.fill_colors[6])
            ctx.rectangle(   0,
                        (image_height / 2.0) + (image_height * rect_size / 2.0),
                        (image_width * rect_size),
                        -1 * (image_height * rect_size)  * self.fill_percentages[6] )

            ctx.fill()
            #empty small rectangles
            ctx.set_source_rgba(*self.background_color)
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
            #fill small rectangles
            #right top
            ctx.set_source_rgba(*self.fill_colors[1])
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[1])
            ctx.fill()
            #right, bottom
            ctx.set_source_rgba(*self.fill_colors[3])
            ctx.rectangle(
                         image_width - image_width * rect_size, 
                         image_height - image_height * rect_size + image_height * rect_size / 3.0,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[3])
            ctx.fill()
            #left bottom
            ctx.set_source_rgba(*self.fill_colors[5])
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height - image_height * rect_size + image_height * rect_size / 3.0,
                        image_width * rect_size / 3.0,
                        -1 *  image_height * rect_size / 3.0 * self.fill_percentages[5])
            ctx.fill()
            #left, top
            ctx.set_source_rgba(*self.fill_colors[7])
            ctx.rectangle(
                         image_width / 2.0 - image_width * rect_size / 2.0, 
                         image_height * rect_size,
                        image_width * rect_size / 3.0,
                        -1 * image_height * rect_size / 3.0 * self.fill_percentages[7])
            ctx.fill()
        ctx.set_source_rgba(*self.border_color)

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
        ctx.set_line_width(image_width / 200.0)
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

        ctx.set_source_rgba(*self.text_colors[0])
        rendered_text( image_width / 2.0, image_height * rect_size / 3.0, 
                      "%s\n%i" % (self.class_names[0],
                                  self.counts[0]
                                 )
                     )
        ctx.set_source_rgba(*self.text_colors[2])
        rendered_text( image_width - image_width * rect_size / 3.0, image_height / 2.0, 
                      "%s\n%i" % (self.class_names[1],
                                 self.counts[2]
                                 )
                     )
        ctx.set_source_rgba(*self.text_colors[4])
        rendered_text( image_width / 2.0, image_height - image_height * rect_size / 3.0, 
                      "%s\n%i" % (self.class_names[2],
                                 self.counts[4]
                                 )
                     )
        ctx.set_source_rgba(*self.text_colors[6])
        rendered_text( image_width * rect_size / 3.0, image_height / 2.0, 
                      "%s\n%i" % (self.class_names[3],
                                 self.counts[6]
                                 )
                     )


        #right top
        ctx.set_source_rgba(*self.text_colors[1])
        rendered_text(image_width - (image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (self.counts[1],))

        ctx.set_source_rgba(*self.text_colors[3])
        rendered_text(image_width - (image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      image_height - (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (self.counts[3],))

        ctx.set_source_rgba(*self.text_colors[5])
        rendered_text((image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      image_height - (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (self.counts[5],))
        #left top
        ctx.set_source_rgba(*self.text_colors[7])
        rendered_text((image_width * rect_size / 2.0 + image_width / 2.0) / 2.0,
                      (image_height * rect_size / 2.0 + image_height / 2.0) / 2.0,
                      "%i" % (self.counts[7],))
            
        if output_filename.endswith('.pdf'):
            pass #taken care of when initializing the surface
        else:
            surface.write_to_png(output_filename)
        surface.finish()
        data = {
            'Condition': [],
            'Count': [],
            'Fill': [],
        }
        for ii in xrange(0, 8, 2):
            data['Condition'].append(self.class_names[ii / 2])
            data['Count'].append(self.counts[ii])
            data['Fill'].append(self.fill_percentages[ii])

        data['Condition'].append(self.class_names[0] + ' and ' + self.class_names[1])
        data['Count'].append(self.counts[1])
        data['Fill'].append(self.fill_percentages[1])

        data['Condition'].append(self.class_names[1] + ' and ' + self.class_names[2])
        data['Count'].append(self.counts[3])
        data['Fill'].append(self.fill_percentages[3])

        data['Condition'].append(self.class_names[2] + ' and ' + self.class_names[3])
        data['Count'].append(self.counts[5])
        data['Fill'].append(self.fill_percentages[5])

        data['Condition'].append(self.class_names[0] + ' and ' + self.class_names[3])
        data['Count'].append(self.counts[7])
        data['Fill'].append(self.fill_percentages[7])
        data['Condition'] = [x.replace("\n", ' ') for x in data['Condition']]
        df = exptools.DF.DataFrame(data)
        exptools.DF.DF2Excel().write(df, output_filename[:output_filename.rfind('.')] + '.xls')


class SquareEuler(SquareEulerFromNumbers):
    """Plot a square pseudo euler diagram of two groups of sets,
    where within a group the sets are non-overlapping"""

    def __init__(self, class_tuples, filled_values = None):
        try:
            self.assert_distinct(class_tuples[0][1], class_tuples[2][1])
        except ValueError:
            raise ValueError("Non-overlapping sets were not distinct! %s %s"%  (class_tuples[0][0], class_tuples[2][0]))
        try:
            self.assert_distinct(class_tuples[1][1], class_tuples[3][1])
        except ValueError:
            raise ValueError("Non-overlapping sets were not distinct! %s %s"%  (class_tuples[1][0], class_tuples[3][0]))
        if len(class_tuples) != 4:
            raise ValueError("Square_Euler takes exactly four tuples of (name, set)")

        names, counts, fill_percentages = self.calculate_counts(class_tuples, filled_values)
        SquareEulerFromNumbers.__init__(self, names, counts, fill_percentages)

    def calculate_counts(self, class_tuples, filled_values):
        sets = [None] * 8
        sets[0] = (class_tuples[0][1] - class_tuples[1][1] - class_tuples[3][1])
        sets[1] = (class_tuples[1][1].intersection(class_tuples[0][1]))
        sets[2] = (class_tuples[1][1] - class_tuples[0][1] - class_tuples[2][1])
        sets[3] = (class_tuples[2][1].intersection(class_tuples[1][1]))
        sets[4] = (class_tuples[2][1] - class_tuples[1][1] - class_tuples[3][1])
        sets[5] = (class_tuples[3][1].intersection(class_tuples[2][1]))
        sets[6] = (class_tuples[3][1] - class_tuples[0][1] - class_tuples[2][1])
        sets[7] = (class_tuples[0][1].intersection(class_tuples[3][1]))
        counts = []
        for s in sets:
            counts.append(len(s))
        names = [
            class_tuples[0][0], 
            class_tuples[1][0],
            class_tuples[2][0],
            class_tuples[3][0],
        ]
        fill_percentages = None
        if filled_values:
            fill_percentages = []
            for s in sets:
                if s:
                    fill_percentages.append(float(len(s.intersection(filled_values))) / len(s))
                else:
                    fill_percentages.append(0)
        else:
            fill_percentages = [1] * len(sets)
        return names, counts, fill_percentages
 
    def assert_distinct(self, setA, setB):
        if len(setA.intersection(setB)) > 0:
            raise ValueError("Non-overlapping sets were not distinct!")

if __name__ == '__main__': 
    x = SquareEulerFromNumbers(["A","B","C","D"],
          [100, 150, 200, 250, 300, 350, 400, 450],
            fill_percentages = [0.25, 0.5, 0.75, 1, 0.2, 0.4, 0.6, 0.8]
    )
    x.fill_colors = [
                (0, 1, 0, 1), #green
                (1,0,0,1), #red
                (0,0,1,1), #blue
                (0,0,0,1), # black,
                (0.8,0.8,0.8,1), # dark grey,
                (1,0,1,1), # purple,
                (0,1,1,1), # green,
                (1,1,0,1), # gold,
        ]
    x.background_color = (1,1,1,1)
    x.text_colors = [
                (1,0,0,1), # blue,
                (0,1,0,1), # green,
                (1,0,0,1), # red,
                (1,1,0.5,1), # white,
                (1,0,0,1), # red,
                (0,1,0,1), # green,
                (0,0,1,1), # blue,
                (1,0,1,1), # purple,

        ]
    x.plot("test.png")
