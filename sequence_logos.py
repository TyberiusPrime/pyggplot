import math
import re
import cairo

def plot_sequences(sequences, output_filename, is_dna = True, width=7, height=2):
    sequences = [s.upper() for s in sequences]
    if output_filename.endswith('.png'):
        sft = 'image'
        width *= 150
        height *= 150
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    elif output_filename.endswith('.pdf'):
        surface = cairo.PDFSurface(width, height)
        sft = 'pdf'
    else:
        raise ValueError("Don't know how to create a surface for this extension: %s" % output_filename)
    ctx = cairo.Context(surface)

    colors = {
        'foreground': (0,0,0,),
        'background': (1,1,1),
    }
    if is_dna:
        colors.update({
            'letter_G': (1,165 / 255.0,0),     #ffa500
            'letter_T': (1,0,0),     #ff0000
            'letter_U': (1,0,0),     #ff0000
            "letter_C": (0,0,1),     #0000ff
            'letter_A': (0,0.5,0),     #008000
        })
        bitsPerPosition = 4
    else:
        colors.update(ProteinColorCode().getColors())
        bitsPerPosition = 20
    drawer = LogoOnCairoDrawer()
    drawer.draw(ctx, sequences, colors, bitsPerPosition, width, height)
    surface.flush()
    if (sft == 'image'):
        op = open(output_filename,'wb')
        surface.write_to_png(op)
        op.close()
    elif sft == 'pdf':
        pass
    surface.finish()

class LogoOnCairoDrawer:

    def draw(self, ctx, sequences, colors, bitsPerPosition, width, height):
        bg = colors['background']
        ctx.set_source_rgb(bg[0],bg[1],bg[2])
        ctx.rectangle(0,0,width,height)
        ctx.fill()
        fg = colors['foreground']
        ctx.set_source_rgb(fg[0],fg[1],fg[2])
        ctx.rectangle(0,0,width,height)
        ctx.stroke()
        ctx.select_font_face("Courier New")        
        scaleX, scaleY = self.findFontScalingFactors(ctx)
        org = ctx.set_matrix(cairo.Matrix(1,0,0,1,0,0))
        if len(sequences) > 0:
            countMatrix = self.getCounts(sequences)
            columnSum = len(sequences)
            lettersToDraw = []
            #transform to sorted lists of (letter, height)
            for column in countMatrix:
                H = 0
                for letter, count in column.items():
                    frequency = float(count) / columnSum
                    H += frequency * math.log(frequency,2)
                H = -H
                print H
                correction = 0
                RSequence = math.log(bitsPerPosition,2) - (H + correction)
                print RSequence
                for letter in column:
                    count = column[letter]
                    frequency = float(count) / columnSum
                    letterHeight = frequency * RSequence
                    column[letter] = letterHeight
                letters = column.items()
                letters.sort(lambda a,b: int(a[1] * 1000 - b[1] * 1000))
                lettersToDraw.append(letters)            
            
            x = 0            
            letterWidth = float(width) / len(lettersToDraw) * 0.9
            heightPerBit = float(height - 10) / math.log(bitsPerPosition,2)             
            pos = 0
            yStart = ctx.device_to_user(0,height)[1] - 5            
            org = ctx.get_matrix() 
            #tempMatrix = cairo.Matrix(letterWidth,0,0,10,0,0)                    
            #ctx.set_font_matrix(tempMatrix)                                
            print lettersToDraw
            for counts in lettersToDraw:                
                pos += 1
                yOffset = 0     
                for char, height in counts:     
                    if 'letter_' + char in colors:
                        rgb = colors['letter_' + char]
                        ctx.set_source_rgb(rgb[0],rgb[1],rgb[2])
                    else:     
                        ctx.set_source_rgb(0,0,0)
                    if height > 0:
                        charHeight = height * heightPerBit    
                        ctx.move_to(x,yStart - yOffset)                                        
                        tempMatrix = cairo.Matrix(letterWidth * scaleX,0,0,charHeight * scaleY,0,0)                    
                        ctx.set_font_matrix(tempMatrix)                    
                        ctx.show_text(char)           
                        yOffset += charHeight                                             
                x += letterWidth * 1.1
                #break
    
    def findFontScalingFactors(self,ctx):
        tempMatrix = cairo.Matrix(100,0,0,100.0,0,0)
        ctx.set_font_matrix(tempMatrix)
        ext = ctx.text_extents("M")        
        width = ext[2]
        height = ext[3]        
        return 100.0 / width, 100.0 / height

    def getCounts(self,sequences):
        lettersByPosition = []        
        for i in xrange(0, len(sequences[0])):
            pos = []
            for s in sequences:                              
                pos.append(s[i])
            lettersByPosition.append(pos)
        countMatrix = []
        for column in lettersByPosition:
           counts = {}
           for c in column:
               if not c in counts:
                   counts[c] = 0
               counts[c] += 1
           countMatrix.append(counts)
        return countMatrix    
     
class ProteinColorCode:
    def getColors(self):
        rasMolColors = """D,E   bright red [230,10,10]     C,M     yellow [230,230,0]
  K,R   blue       [20,90,255]     S,T     orange [250,150,0]
  F,Y   mid blue   [50,50,170]     N,Q     cyan   [0,220,220]
  G       light grey [235,235,235]   L,V,I green  [15,130,15]
  A       dark grey  [200,200,200]   W         pink   [1G0,90,180]
  H       pale blue  [130,130,210]   P         flesh  [220,150,130]"""        
        colors = re.findall('([A-Z,]+)[\sa-z]+\[([\d,]+)\]',rasMolColors)
        res = {}
        for aas, color in colors:
            color = color.split(',')
            color = [float(c) / 255 for c in color]
            for oneletter in aas.split(","):
                res["letter_" + oneletter.upper()] = color
        return res
                


