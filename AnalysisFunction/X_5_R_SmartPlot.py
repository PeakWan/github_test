# @Time : 2020-11-19
# V 1.0.0
# @Author : xuyuan

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def R_froest_plot(df_input,title,name,mean,lowerb,upperb,savePath,grawid=5,tickfont=1, xlabfont=1, style=1,zero=1):
    forest=ro.r('''
          #V1.0.4
#Author shanqingsong
#date 2020-12-10

#V1.0.0：第1版forestplot
#V1.0.3: 名称增加随机数
#v1.0.4:增加1种绘图风格，优化绘图问题
myforestplot <- function(dataset,title,name,mean,lowerb,upperb,savePath,grawid=5,tickfont=1, xlabfont=1, style=1,zero=1){
  # input:  
  # dataset ：dataframe 数据,
  # title: 森林图标题
  # name: 名称列
  # mean: str or值
  # lowerb:str or估计下界
  # upperb:str or估计上界
  # savePath:str 存储路径
  # grawid: int 图形宽度(cm)
  # tickfont:int 刻度字体
  # xlabfont：int xlab字体 
  # return: 
  # forestname:strVector 图片保存位置
  library(dplyr)
  library(forestplot)
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:100,1)
  dataset <- as.data.frame(dataset) 
  for(i in 1:nrow(dataset)){
    if(sum(is.na(c(dataset[,mean][i],dataset[,lowerb][i],dataset[,upperb][i])))!=0) {
      dataset$newcol[i] <- NA
      next}
    dataset$newcol[i] <- paste0(dataset[,mean][i],"(",dataset[,lowerb][i],",",dataset[,upperb][i],")")
    print(i)
  }
  dataset$newcol %>% print()
  tabletext <- cbind(dataset[,name]%>% as.character(),dataset$newcol)
  print(tabletext)
  forestname <- paste0("Forest",mytime,"_",rand,".jpeg")
  style1=fpColors(box="black",line="grey", summary="grey",zero = 'grey')
  style2=fpColors(box="royalblue",line="darkblue", summary="royalblue")
  jpeg(file=paste0(savePath,forestname),width=6,height=8,units="in",res=300)
  if(style==1){
    forestplot(tabletext, 
               title = title,
               mean = dataset[,mean],
               lower = dataset[,lowerb],
               upper = dataset[,upperb],
               new_page = TRUE,
               boxsize = 0.15,
               # is.summary=c(TRUE,TRUE,rep(FALSE,8),TRUE),
               # clip=c(0.1,2.5), 
               xlog=FALSE, 
               align = 'r',#优化1:变量名称过长问题
               zero = zero,
               col=style1)
  }else{
    forestplot(tabletext, 
               title = title,
               mean = dataset[,mean],
               lower = dataset[,lowerb],
               upper = dataset[,upperb],
               new_page = TRUE,
               boxsize = 0.15,
               # is.summary=c(TRUE,TRUE,rep(FALSE,8),TRUE),
               # clip=c(0.1,2.5), 
               xlog=FALSE, 
               align = 'r',#优化1:变量名称过长问题
               zero =zero,
               col=style2)
  }
  dev.off()
  return(forestname)
}
        ''')

    df_input[[mean,lowerb,upperb]]= df_input[[mean,lowerb,upperb]].astype(float)
    df_input[[name]]= df_input[[name]].astype(str)
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_input)
    result=forest(r_df,title,name,mean,lowerb,upperb,savePath,grawid,tickfont, xlabfont, style,zero)
    result_path=tuple(result)[0]
    return result_path