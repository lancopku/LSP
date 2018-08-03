# Latent Structured Perceptron Toolkit v1.0
This is a general purpose software for sequential tagging with the emphasis on fast training speed. This toolkit includes Latent Structured Perceptron (LSP) model (Sun et al., IJCAI 2009, TKDE 2013). It also includes traditional Structured Perceptron (SP) model and with the averaged version (Collins, 2002). [[Tutorial]](LSP.tu.pdf)

Main features:

 - Developed with C#
 - Automatic modeling of hidden information (latent structures) in the data (Sun et al., IJCAI 2009, TKDE 2013)
 - Fast training (much faster than probabilistic models like CRFs)
 - General purpose (it is task-independent & trainable using your own tagged corpus)
 - Support rich edge features (Sun et al., ACL 2012)
 - Support various evaluation metrics, including token-accuracy, string-accuracy, & F-score
