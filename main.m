[A,B,C,D,E] = textread('iris.data','%f%f%f%f%s','delimiter',',');
Group=unique(E);
targets=zeros(size(E));
for i=1:size(E,1) %��string���ת�����������
    if isequal(E(i),Group(1))
        targets(i)=1;
    elseif isequal(E(i),Group(2))
        targets(i)=2;
    elseif isequal(E(i),Group(3))
        targets(i)=3;
    end
end
inputdata=[A,B,C,D,targets];
indices = crossvalind('Kfold',size(inputdata,1),10); %����ʮ�۽�����֤��ǩ
for i = 1:10
    test = (indices == i); train = ~test;
    traindata=inputdata(train,:);
    testdata=inputdata(test,:);
    %���ݱ�ǩȷ����ǰѵ������������֤������
    train_patterns=traindata(:,1:(size(traindata,2)-1));  
    train_targets=traindata(:,size(traindata,2))';  
    test_patterns=testdata(:,1:(size(testdata,2)-1));  
    test_targets=testdata(:,size(testdata,2))';  
    test_targets_predict = C4_5(train_patterns', train_targets, test_patterns', 5, 10);  
    %train_patterns'����feature����������  
    %train_targets ��1�ж��У�����ѵ����������  
    % test_patterns'����feature����������  
    %�����������ȡֵ����C4_5����  
    temp_count=0;  
    for i=1:size(test_targets_predict,2)  
        if test_targets(:,i)==test_targets_predict(:,i)  
            temp_count=temp_count+1;  
        end  
    end  
    accuracy=temp_count/size(test_targets,2);  
    disp(accuracy); 
end
