#library(tidyverse)
LogReg <- function(mydata,target, feature_con,feature_cat, interaction = NULL,cate=NULL, cat_ref_lev=NULL,method="ALL", nomogram = "yes",savePath,round= 2){
  # 定义变量
  library(MASS)
  library(pROC)
  library(rms)
  library(dplyr)
  library(Cairo)
  results <- list() #结果
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  # 数据预处理
  mydata[,feature_cat] <- lapply(mydata[,feature_cat], factor)
  mydata <- data.frame(mydata[feature_cat],mydata[feature_con],mydata[target])
  mydata <- na.omit(mydata) # 除去NA
  if(length(cate)>0){
    for (i in 1:length(cate)){
      mydata[,cate[i]] <-  relevel(mydata[,cate[i]], ref = cat_ref_lev[i])
    }
  }
  # feature_con 连续变量
  # feature_cat 类别变量
  # interaction 交互项
  # cate 需要指定参考标签的类别变量
  feature <- c(feature_con,feature_cat, interaction)
  # setwd(savePath)
  # 建立模型
  formula <- paste0(target,'~',paste(feature,collapse = '+'))
  # logistic 模型
  fit <- glm(as.formula(formula),data=mydata,family = binomial())
  # 变量选择。 outputfit为输出模型
  # 将下面改为switch
  if (method == "forward"){
    outputfit <- step(fit,direction = 'forward')
  } else
    if (method == "backward"){
      outputfit <- step(fit,direction = 'backward')
    } else
      if(method == "both"){
        outputfit <- step(fit,direction = 'both')
      } else{
        outputfit <- fit
      }
  # 输出系数及or值的整理
  results$model <- outputfit
  pvalue <- summary(outputfit)$coefficient[,4]
  coefs <- cbind(coef(outputfit), confint(outputfit), exp(coef(outputfit)), exp(confint(outputfit)),pvalue)
  coefs <- round(coefs,round)
  rownames(coefs)  <- NULL
  colnames(coefs) <- c("系数","系数下限","系数上限","OR","OR下限","OR上限","pvalue")
  coefs <- data.frame(name = names(outputfit$coefficients), coefs)
  #print(coef)
  results$coefs <- coefs
  # 画nomogram， 想法是从最优模型中得到formula，代入lrm命令中
  if(nomogram=="yes"){
    ddist <<- datadist(mydata)
    options(datadist='ddist')
    tempfit <- lrm(formula(outputfit),data=mydata)
    nom <- nomogram(tempfit, fun=plogis,
                    fun.at=c(.001, .01, .05, seq(.1,.9, by=.1), .95, .99, .999),
                    lp=F, funlabel="Risk")
    Nomoname <- paste0("Nomogram",mytime,".jpeg")
    CairoJPEG(file=paste0(savePath,Nomoname),width=5,height=5,units="in",res=600)
    plot(nom)
    dev.off()
    results$nomo <- Nomoname
  }
  # 画ROC曲线
  gfit <- roc(as.formula(paste0(target,'~predict(outputfit)')), data = mydata)
  ROCname <- paste0("ROC",mytime,".jpeg")
  CairoJPEG(file=paste0(savePath,ROCname),width=5,height=5,units="in",res=600)
  plot.roc(gfit,col="red",print.auc =TRUE,print.auc.col = "darkgreen",auc.polygon = TRUE,auc.polygon.col = "pink")
  dev.off()
  #输出ROC数据
  #xcoor <- seq(from = 0,to = 1,length.out = 50)
  #coordi <- coords(gfit, x = xcoor, transpose = FALSE)
  #results$coord <- data.frame("sensitivity" = coordi$sensitivity, "specificity" = coordi$specificity, "Youden" = coordi$sensitivity + coordi$specificity -1)
  #results$coord <- results$coord %>% round(round)
  results$roc <- ROCname
  results$data<- (mydata %>% predict(outputfit, ., type="response") %>% mutate(mydata,.))
  return(results)
}