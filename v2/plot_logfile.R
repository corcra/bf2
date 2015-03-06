library(ggplot2)
library(reshape2)

args<-commandArgs(TRUE)
fname<-args[1]
da<-read.table(fname,header=TRUE)
energies<-da[,c("n","data_energy","model_energy","valiset_energy","random_energy")]
#energies<-da[,c("n","de","me","ve","re")]
dll<-da[,c("n","ll")]

datac<-"deepskyblue"
modelc<-"springgreen3"
valic<-"black"
randomc<-"gray50"

el<-melt(energies,id="n")
ggplot(el, aes(x=n, y=value, color=variable))+geom_point(cex=1.2, alpha=0.8)+xlab("# training examples seen")+ylab("mean energy of batch")+ggtitle(fname)+scale_color_manual(values=c(datac, modelc, valic, randomc))
ggsave(paste0(fname,"_energies.pdf"))

ggplot(dll, aes(x=n, y=ll))+geom_point()
ggsave(paste0(fname,"_ll.pdf"))
