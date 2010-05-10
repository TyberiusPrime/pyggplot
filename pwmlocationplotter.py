import exptools
import math
import cairo
import chipseq

class PWMLocationPlotter:

    def __init__(self, sequences,  pwms_dict, arbritrary_hits = None, sequence_classes = None):
        self.hits = self.pwms_to_hits(sequences, pwms_dict)
        if arbritrary_hits:
            self.hits.update(arbritrary_hits)
        self.base_pairs = max((len(x) for x in sequences))
        self.sequences = sequences
        if sequence_classes is None:
            self.seq_class = [0] * len(sequences)
        else:
            self.seq_class = sequence_classes
        self.load_colors()

    def load_colors(self):
        self.colors = {
            'text': (0,0,0),
            'foreground0': (0.5,0.2,0,),
            'foreground1': (0,0.2,0.5,),
            'background': (1,1,1),
        }
        self.color_scheme = [
            (1,0,0,0.9),
            (0,0,1,0.9),
            (0,1,0,0.9),
            (1,1,0,0.9),
            (0,1,1,0.9)]
        i = 0
        for name in self.hits:
            self.colors[name] = self.color_scheme[i]
            i += 1

    def plot(self, output_filename):
        base_pairs = self.base_pairs
        sequences = self.sequences 
        seq_width = base_pairs / 25.0
        seq_height = len(sequences) / 3.0 
        img_width_inches = seq_width
        legend_height = 1
        img_height_inches = seq_height + legend_height
        dpi = 75.0
        width = int(math.ceil(dpi * img_width_inches))
        height = int(math.ceil(dpi * img_height_inches))
        graph_height = (img_height_inches - legend_height) * dpi
        matrix = cairo.Matrix(width / img_width_inches, 0,0, height / img_height_inches, 0 ,0)
        if output_filename.endswith('.png'):
            sft = 'image'
            surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
        elif output_filename.endswith('.pdf'):
            surface = cairo.PDFSurface(output_filename, width, height)
            sft = 'pdf'
        else:
            raise ValueError("Don't know how to create a surface for this extension: %s" % output_filename)
        ctx = cairo.Context(surface)
        ctx.set_matrix(matrix)

        ctx.set_source_rgb(*self.colors['background'])
        ctx.rectangle(0,0, img_width_inches, img_height_inches)#base_pairs, len(sequences))
        ctx.fill()

        ctx.set_matrix(cairo.Matrix(width / float(base_pairs), 0,0, 
                                    (graph_height) / float(len(sequences)), 0, -1))

        self.plot_sequences(ctx)
        self.plot_features(ctx)
        ctx.set_matrix(matrix)
        self.plot_legend(ctx, img_width_inches, img_height_inches)

        if (sft == 'image'):
            op = open(output_filename,'wb')
            surface.write_to_png(op)
            op.close()
        elif sft == 'pdf':
            pass
        surface.finish()

    def plot_legend(self, ctx, img_width_inches, img_height_inches):
        ctx.set_source_rgb(*self.colors['text'])
        ctx.set_line_width(0.01)
        ctx.rectangle(0, img_height_inches -0.8, img_width_inches, 0)
        ctx.stroke()
        x = 0.2
        ctx.set_font_size(0.3)
        for feature_name in self.hits:
            ctx.set_source_rgba(*self.colors[feature_name])
            ctx.rectangle(x - 0.1, img_height_inches - 0.5, 0.2, 0.2)
            ctx.fill()
            ctx.move_to(x + 0.2, img_height_inches - 0.3)
            ctx.set_source_rgb(*self.colors['text'])
            ctx.show_text(feature_name)
            x += ctx.text_extents(feature_name)[4] + 0.2 + 0.2
            ctx.fill()

    def plot_sequences(self, ctx):
        ctx.set_line_width(0.1)
        #ctx.set_source_rgb(*colors['foreground'])
        last_class = 0
        ctx.set_source_rgb(*self.colors['foreground0'])
        for ii in xrange(0, len(self.sequences)):
            if last_class != self.seq_class[ii]:
                ctx.stroke()
                ctx.set_source_rgb(*self.colors['foreground%i' % self.seq_class[ii]])
                last_class = self.seq_class[ii]
            ctx.move_to(0, ii + 0.9)
            ctx.line_to(len(self.sequences[ii]), ii + 0.9)
        ctx.stroke()

        
    def plot_features(self, ctx):
        ctx.set_line_width(1)
        one_feature_heigth = 0.9 / len(self.hits)
        ctx.set_source_rgba(self.colors['text'][0],self.colors['text'][1],self.colors['text'][2], 0.3)
        for ii, feature_name in enumerate(self.hits):
            for seq_no, bp_pos in self.hits[feature_name]:
                ctx.move_to(bp_pos, seq_no)
                ctx.line_to(bp_pos, seq_no + 0.9)
        ctx.stroke()
        for ii, feature_name in enumerate(self.hits):
            ctx.set_source_rgba(*self.colors[feature_name])
            for seq_no, bp_pos in self.hits[feature_name]:
                #ctx.move_to(bp_pos, seq_no)
                #ctx.line_to(bp_pos, seq_no + 0.9)
                ctx.move_to(bp_pos, seq_no + ii * one_feature_heigth)
                ctx.line_to(bp_pos, seq_no + (1 + ii )* one_feature_heigth)
            ctx.stroke()

    def pwms_to_hits(self, sequences, pwms_dict):
        res = {}
        for pwm_name in pwms_dict:
            res[pwm_name] = []
            pwm = chipseq.pwm.PWM(pwms_dict[pwm_name][0])
            threshold = pwms_dict[pwm_name][1] 
            for ii, seq in enumerate(sequences):
                startpoints, endpoints, scores, best_score, cum_score = pwm.scan(seq, threshold)
                for bp in startpoints:
                    res[pwm_name].append((ii, bp))
        return res

if __name__ == '__main__': 
    import TAMO.MotifTools as MotifTools
    motif = MotifTools.Motif_from_text("AGA")
    p = PWMLocationPlotter(
        ["A" * 250 + "AGA" + "A" * 400,
         "C" * 50,
         "G" * 300,], 
        {'Tetest': (motif, motif.maxscore * 0.5)}, sequence_classes = [0,0,1], arbritrary_hits = {'Summit': [(0, 400), (1, 25), (2, 150)]} )
    p.plot('test.png')



