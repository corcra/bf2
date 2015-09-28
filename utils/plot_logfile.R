library(ggplot2)
library(reshape2)

args<-commandArgs(TRUE)
fname_raw<-args[1]
da<-read.table(fname_raw,header=TRUE, na.strings=c("nan"))
fname<-gsub("_logfile.txt", "", fname_raw)
da<-na.omit(da)

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
#ggsave(paste0(fname,"_energies.pdf"))
ggsave(paste0(fname,"_energies.png"))

# log likelihood
dll<-da[,c("n","ll")]
if (sd(dll$ll) > 0){
    ggplot(dll, aes(x=n, y=ll))+geom_point()+ggtitle(fname)
    #ggsave(paste0(fname,"_ll.pdf"))
    ggsave(paste0(fname,"_ll.png"))
} else{
    print("not plotting log-likelihood (variance 0, probably not recorded)")
}

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
#ggsave(paste0(fname,"_lens.pdf"))
ggsave(paste0(fname,"_lens.png"))

# some evaluation metrics, now
# NOTE: these files may not exist, in which this will fail... but that's ok
# derp
# # accuracies
dev_acc<-read.table(paste0(fname,"_dev_acc.txt"), header=T)
ggplot(dev_acc, aes(x=epoch, y=accuracy, group=scoretype, colour=scoretype))+geom_point(cex=1.2)+geom_line()+xlab("# epochs")+ylab("accuracy on socher task")+ggtitle(fname)
ggsave(paste0(fname,"_dev_acc.png"))

# hits at N
hits_atN<-read.table(paste0(fname,"_hits_at_N.txt"), header=T)
ggplot(hits_atN, aes(x=epoch, y=mean_hits, group=N, colour=N))+geom_point(cex=1.2)+geom_line()+xlab("# epochs")+ylab("fraction true R is in top N")+ggtitle(fname)
ggsave(paste0(fname, "_hits_at_N.png"))

# predictive
predictive<-read.table(paste0(fname,"_predictive.txt"), header=T)
ggplot(predictive, aes(x=epoch, y=mean_pRAC))+geom_point(cex=1.2)+geom_line()+xlab("# epochs")+ylab("mean p(R|A, C) for true triples")+ggtitle(fname)
ggsave(paste0(fname, "_predictive.png"))

# AUC
auc<-read.table(paste0(fname, "_auc.txt"), header=T)
ggplot(auc, aes(x=epoch, y=auc, group=scoretype, colour=scoretype))+geom_point(cex=1.2)+geom_line()+xlab("# epochs")+ylab("AUROC")+ggtitle(fname)
ggsave(paste0(fname, "_auc.png"))
