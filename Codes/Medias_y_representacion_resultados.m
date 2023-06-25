clc
clear all 
close all 

tbl1 = readtable('PrimerTraining.csv');

disp(tbl1)
mean_overlapping_bboxes = mean(tbl1.(1))
class_acc1 = mean(tbl1.(2))
loss_rpn_cls1 = mean(tbl1.(3))
loss_rpn_regr1 = mean(tbl1.(4))
loss_class_cls1 = mean(tbl1.(5))
loss_class_regr1 = mean(tbl1.(6))
curr_loss1 = mean(tbl1.(7))
elapsed_time1 = mean(tbl1.(8))
mAP1 = mean(tbl1.(9))

numRows = height(tbl1);
disp(numRows)
x = 0:1:numRows-1;

%---REPRESENTACIÓN DE LOS VALORES DE LOS LOSSES--------
f1 = figure(1);

plot(x,tbl1.(3),'r')
title('\textbf{loss rpn cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f1, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/LossRpnCls1.pdf")

f2 = figure(2);
plot(x,tbl1.(4),'r')
title('\textbf{loss rpn regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f2, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/LossRpnRegr1.pdf")

f3 = figure(3);
plot(x,tbl1.(5),'r')
title('\textbf{loss class cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f3, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/LossClassCls1.pdf")

f4 = figure(4);
plot(x,tbl1.(6),'r')
title('\textbf{loss class regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f4, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/LossClassRegr1.pdf")
%---REPRESENTACIÓN DEL RESTO DE VALORES---%

f5 = figure(5);
plot(x,tbl1.(7),'r')
title('\textbf{curr loss}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f5, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/CurrLoss1.pdf")

f6 = figure(6);
plot(x,tbl1.(1),'b')
title('\textbf{mean overlapping boxes}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f6, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/MeanOverlappingBoxes1.pdf")

f7 = figure(7);
plot(x,tbl1.(2),'b')
title('\textbf{Class Accuracy}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f7, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/ClassAccuracy1.pdf")

f8 = figure(8);
plot(x,tbl1.(8),'b')
title('\textbf{Tiempo/\textit{epoch} transcurrido (s)}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{tiempo (s)}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f8, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/PrimerEntrenamiento/Tiempo1.pdf")

%%
clc
clear all 
close all 

tbl2 = readtable('SegundoTraining.csv');

disp(tbl2)
mean_overlapping_bboxes2 = mean(tbl2.(1))
class_acc2 = mean(tbl2.(2))
loss_rpn_cls2 = mean(tbl2.(3))
loss_rpn_regr2 = mean(tbl2.(4))
loss_class_cls2 = mean(tbl2.(5))
loss_class_regr2 = mean(tbl2.(6))
curr_loss2 = mean(tbl2.(7))
elapsed_time2 = mean(tbl2.(8))
mAP2 = mean(tbl2.(9))

numRows = height(tbl2);
disp(numRows)
x = 0:1:numRows-1;

%---REPRESENTACIÓN DE LOS VALORES DE LOS LOSSES--------
f1 = figure(1);

plot(x,tbl2.(3),'r')
title('\textbf{loss rpn cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f1, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/LossRpnCls2.pdf")

f2 = figure(2);
plot(x,tbl2.(4),'r')
title('\textbf{loss rpn regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f2, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/LossRpnRegr2.pdf")

f3 = figure(3);
plot(x,tbl2.(5),'r')
title('\textbf{loss class cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f3, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/LossClassCls2.pdf")

f4 = figure(4);
plot(x,tbl2.(6),'r')
title('\textbf{loss class regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f4, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/LossClassRegr2.pdf")
%---REPRESENTACIÓN DEL RESTO DE VALORES---%

f5 = figure(5);
plot(x,tbl2.(7),'r')
title('\textbf{curr loss}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f5, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/CurrLoss2.pdf")

f6 = figure(6);
plot(x,tbl2.(1),'b')
title('\textbf{mean overlapping boxes}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f6, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/MeanOverlappingBoxes2.pdf")

f7 = figure(7);
plot(x,tbl2.(2),'b')
title('\textbf{Class Accuracy}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f7, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/ClassAccuracy2.pdf")

f8 = figure(8);
plot(x,tbl2.(8),'b')
title('\textbf{Tiempo/\textit{epoch} transcurrido (s)}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{tiempo (s)}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f8, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/SegundoEntrenamiento/Tiempo2.pdf")
%%
clc
clear all 
close all 

tbl3 = readtable('TercerTraining.csv');

disp(tbl3)
mean_overlapping_bboxe3 = mean(tbl3.(1))
class_acc3 = mean(tbl3.(2))
loss_rpn_cls3 = mean(tbl3.(3))
loss_rpn_regr3 = mean(tbl3.(4))
loss_class_cls3 = mean(tbl3.(5))
loss_class_regr3 = mean(tbl3.(6))
curr_loss3 = mean(tbl3.(7))
elapsed_time3 = mean(tbl3.(8))
mAP3 = mean(tbl3.(9))

numRows = height(tbl3);
disp(numRows)
x = 0:1:numRows-1;

%---REPRESENTACIÓN DE LOS VALORES DE LOS LOSSES--------
f1 = figure(1);

plot(x,tbl3.(3),'r')
title('\textbf{loss rpn cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f1, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/LossRpnCls3.pdf")

f2 = figure(2);
plot(x,tbl3.(4),'r')
title('\textbf{loss rpn regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f2, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/LossRpnRegr3.pdf")

f3 = figure(3);
plot(x,tbl3.(5),'r')
title('\textbf{loss class cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f3, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/LossClassCls3.pdf")

f4 = figure(4);
plot(x,tbl3.(6),'r')
title('\textbf{loss class regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f4, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/LossClassRegr3.pdf")
%---REPRESENTACIÓN DEL RESTO DE VALORES---%

f5 = figure(5);
plot(x,tbl3.(7),'r')
title('\textbf{curr loss}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f5, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/CurrLoss3.pdf")

f6 = figure(6);
plot(x,tbl3.(1),'b')
title('\textbf{mean overlapping boxes}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f6, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/MeanOverlappingBoxes3.pdf")

f7 = figure(7);
plot(x,tbl3.(2),'b')
title('\textbf{Class Accuracy}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f7, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/ClassAccuracy3.pdf")

f8 = figure(8);
plot(x,tbl3.(8),'b')
title('\textbf{Tiempo/\textit{epoch} transcurrido (s)}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{tiempo (s)}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f8, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/TercerEntrenamiento/Tiempo3.pdf")

%%
clc
clear all 
close all 

tbl4 = readtable('CuartoTraining.csv');

disp(tbl4)
mean_overlapping_bboxes4 = mean(tbl4.(1))
class_acc4 = mean(tbl4.(2))
loss_rpn_cls4 = mean(tbl4.(3))
loss_rpn_regr4 = mean(tbl4.(4))
loss_class_cls4 = mean(tbl4.(5))
loss_class_regr4 = mean(tbl4.(6))
curr_loss4 = mean(tbl4.(7))
elapsed_time4 = mean(tbl4.(8))
mAP4 = mean(tbl4.(9))

numRows = height(tbl4);
disp(numRows)
x = 0:1:numRows-1;

%---REPRESENTACIÓN DE LOS VALORES DE LOS LOSSES--------
f1 = figure(1);

plot(x,tbl4.(3),'r')
title('\textbf{loss rpn cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f1, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/LossRpnCls4.pdf")

f2 = figure(2);
plot(x,tbl4.(4),'r')
title('\textbf{loss rpn regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f2, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/LossRpnRegr4.pdf")

f3 = figure(3);
plot(x,tbl4.(5),'r')
title('\textbf{loss class cls}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f3, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/LossClassCls4.pdf")

f4 = figure(4);
plot(x,tbl4.(6),'r')
title('\textbf{loss class regr}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f4, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/LossClassRegr4.pdf")
%---REPRESENTACIÓN DEL RESTO DE VALORES---%

f5 = figure(5);
plot(x,tbl4.(7),'r')
title('\textbf{curr loss}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{\textit{loss value}}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f5, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/CurrLoss4.pdf")

f6 = figure(6);
plot(x,tbl4.(1),'b')
title('\textbf{mean overlapping boxes}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f6, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/MeanOverlappingBoxes4.pdf")

f7 = figure(7);
plot(x,tbl4.(2),'b')
title('\textbf{Class Accuracy}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\mathbf{\#_{bboxes}}$','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f7, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/ClassAccuracy4.pdf")

f8 = figure(8);
plot(x,tbl4.(8),'b')
title('\textbf{Tiempo/\textit{epoch} transcurrido (s)}','Interpreter','latex','fontweight','bold','fontsize',18)
xlabel('\textbf{\textit{epochs}}','Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('\textbf{tiempo (s)}','Interpreter','latex','fontweight','bold','fontsize',15)
exportgraphics(f8, "/Users/pablosreyero/Desktop/MatlabFiguresTFG/CuartoEntrenamiento/Tiempo4.pdf")
