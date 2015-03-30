library(ggplot2)
library(reshape2)

args<-commandArgs(TRUE)
fname<-args[1]
da<-read.table(fname,header=TRUE, na.strings=c("nan"))
da<-na.omit(da)
print(da[1,])

datac<-"deepskyblue"
modelc<-"springgreen3"
valic<-"black"
randomc<-"gray50"
permc<-"tomato1"

# energies
if (da$model_energy[1] == "NA"){
    energies<-da[,c("n","data_energy","valiset_energy","random_energy", "perm_energy")]
    colz<-c(datac, valic, randomc, permc)
} else{
    energies<-da[,c("n","data_energy","model_energy","valiset_energy","random_energy", "perm_energy")]
    colz<-c(datac, modelc, valic, randomc, permc)
}
el<-melt(energies,id="n")
ggplot(el, aes(x=n, y=value, color=variable))+geom_point(cex=1.2, alpha=0.8)+xlab("# training examples seen")+ylab("mean energy of batch")+ggtitle(fname)+scale_color_manual(values=colz)
#+ylim(-100,10)
ggsave(paste0(fname,"_energies.pdf"))

# log likelihood
dll<-da[,c("n","ll")]
ggplot(dll, aes(x=n, y=ll))+geom_point()+ggtitle(fname)
ggsave(paste0(fname,"_ll.pdf"))

# compare ll with weirdratio
#ratio<-abs(da$valiset_energy/da$perm_energy-(da$valiset_energy/da$perm_energy)[1])
#ratio<-ratio/max(ratio)
#lln<-abs(da$ll-da$ll[1])
#lln<-lln/max(lln)
#dlr<-data.frame(da$n, ratio, lln)
#names(dlr)<-c("n","ratio","abs(ll)")
#dlrl<-melt(dlr, id="n")
#ggplot(dlrl, aes(x=n, y=value, color=variable))+geom_point()+xlab("# training examples seen")+ylab("rescaled measure")+ggtitle(fname)
#ggsave(paste0(fname,"_rtest.pdf"))

# look at vector lengths innit
lens<-da[,c("n", "C_lens", "G_lens", "V_lens")]
lensl<-melt(lens, id="n")
ggplot(lensl, aes(x=n, y=value, color=variable))+geom_point(cex=1.2, alpha=0.8)+xlab("# training examples seen")+ylab("mean length of sample of vectors/matrices")+ggtitle(fname)
ggsave(paste0(fname,"_lens.pdf"))
