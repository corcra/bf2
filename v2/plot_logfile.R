library(ggplot2)
library(reshape2)

args<-commandArgs(TRUE)
fname<-args[1]
da<-read.table(fname,header=TRUE)

datac<-"deepskyblue"
modelc<-"springgreen3"
valic<-"black"
randomc<-"gray50"
permc<-"tomato1"
colz<-c(datac, modelc, valic, randomc, permc)

# energies
energies<-da[,c("n","data_energy","model_energy","valiset_energy","random_energy", "perm_energy")]
#energies<-da[,c("n","de","me","ve","re")]
el<-melt(energies,id="n")
ggplot(el, aes(x=n, y=value, color=variable))+geom_point(cex=1.2, alpha=0.8)+xlab("# training examples seen")+ylab("mean energy of batch")+ggtitle(fname)+scale_color_manual(values=colz)
ggsave(paste0(fname,"_energies.pdf"))

# log likelihood
dll<-da[,c("n","ll")]
ggplot(dll, aes(x=n, y=ll))+geom_point()+ggtitle(fname)
ggsave(paste0(fname,"_ll.pdf"))

# compare ll with weirdratio
ratio<-abs(da$valiset_energy/da$perm_energy-(da$valiset_energy/da$perm_energy)[1])
ratio<-ratio/max(ratio)
lln<-abs(da$ll-da$ll[1])
lln<-lln/max(lln)
dlr<-data.frame(da$n, ratio, lln)
names(dlr)<-c("n","ratio","abs(ll)")
dlrl<-melt(dlr, id="n")
ggplot(dlrl, aes(x=n, y=value, color=variable))+geom_point()+xlab("# training examples seen")+ylab("rescaled measure")+ggtitle(fname)
ggsave(paste0(fname,"_rtest.pdf"))
