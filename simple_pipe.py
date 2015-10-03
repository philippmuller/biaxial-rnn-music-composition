from midi_to_statematrix import *
import numpy
import multi_training
import model

name="what"

pcs = multi_training.loadPieces("music")

xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))



# print xIpt
all_outputs = [xOpt[0]]
all_inputs = [xIpt[0]]
print pcs

noteStateMatrixToMidi(numpy.array(pcs['short']),'output/'+name)