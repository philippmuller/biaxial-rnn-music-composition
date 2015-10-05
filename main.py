import cPickle as pickle
import gzip
import numpy
from midi_to_statematrix import *
import multi_training
import model
import daemon
import lockfile

# initialize daemon
context = daemon.DaemonContext(
  working_directory="/home/azureuser/hackzurich",
  umask=0o002,
  pidfile=lockfile.FileLock("/var/run/musicrnn.pid")
  )

mail_gid = grp.getgrnam("mail").gr_gid
context.gid = mail_gid

parameters = open("output/params0.p")
sample0 = open("output/sample0.mid")
out = open("output/final_learned_config.p")
mid = open("output/final.mid")

context.files_preserve = [parameters, sample0, out, mid]

# generate
def gen_adaptive(m,pcs,times,name="final"):
  xIpt, xOpt = map(lambda x: numpy.array(x, dtype="int8"), multi_training.getPieceSegment(pcs))
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

  noteStateMatrixToMidi(numpy.array(all_outputs),"/home/azureuser/hackzurich/output/"+name)


def create_model():
   return model.Model([300,300],[100,50], dropout=0.5) # [300,300],[100,50]

def create_pieces():
  return multi_training.loadPieces("music")

def web_endpoint_create():
  pcs = multi_training.loadPieces("music")
  m = create_model()
  return m,pcs

def web_endpoint(m, pcs):
  return gen_adaptive(m, pcs, 1, name="live")




#learn
with context:
  pcs = multi_training.loadPieces("more_music")
  print "--> loaded pieces"
  m = create_model()
  print "--> created model"
  multi_training.trainPiece(m, pcs, 1000)
  # 10000
  print "--> training finished"

  pickle.dump( m.learned_config, out )
  gen_adaptive(m, pcs, 2, name="trained")



# if __name__ == "__main__":

#   pcs = multi_training.loadPieces("more_music")
#   print "--> loaded pieces"
#   m = create_model()
#   print "--> created model"
#   multi_training.trainPiece(m, pcs, 100)
#   # 10000
#   print "--> training finished"

#   pickle.dump( m.learned_config, open( "output/final_learned_config.p", "wb" ) )
#   gen_adaptive(m, pcs, 2, name="trained")
