import cPickle as pickle
import gzip
import numpy
from midi_to_statematrix import *

import multi_training
import model

def gen_adaptive(m,pcs,times,name="final"):
	xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))
	all_outputs = [xOpt[0]]
	m.start_slow_walk(xIpt[0])
	cons = 1
	for time in range(multi_training.batch_len*times):
		resdata = m.slow_walk_fun( cons )
		nnotes = numpy.sum(resdata[-1][:,0])
		if nnotes < 2:
			if cons > 1:
				cons = 1
			cons -= 0.02
		else:
			cons += (1 - cons)*0.3

		all_outputs.append(resdata[-1])

	noteStateMatrixToMidi(numpy.array(all_outputs),'output/'+name)


if __name__ == '__main__':

	pcs = multi_training.loadPieces("music")
	print "--> loaded pieces"
	m = model.Model([3,3],[10,5], dropout=0.3)
	# [300,300],[100,50]
	print "--> created model"
	multi_training.trainPiece(m, pcs, 10)
	# 10000
	print "--> training finished"

	pickle.dump( m.learned_config, open( "output/final_learned_config.p", "wb" ) )
