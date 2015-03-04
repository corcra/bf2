library(ggplot2)
library(reshape2)

args<-commandArgs(TRUE)
fname<-args[1]
da<-read.table(fname,header=TRUE)
energies<-da[,c("n","de","me","ve","re")]
dll<-da[,c("n","ll")]

el<-melt(energies,id="n")
ggplot(el, aes(x=n, y=value, color=variable))+geom_point()+xlab("# training examples seen")+ylab("mean energy of batch")+ggtitle(fname)
ggsave(paste0(fname,"_energies.pdf"))

ggplot(dll, aes(x=n, y=ll))+geom_point()
ggsave(paste0(fname,"_ll.pdf"))
