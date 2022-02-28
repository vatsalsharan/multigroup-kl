load('mnist_digits.mat')
digits_0 = digits_0(:,1:1200);
digits_1 = digits_1(:,1:1200);
digits_2 = digits_2(:,1:1200);
digits_3 = digits_3(:,1:1200);
digits_4 = digits_4(:,1:1200);
digits_5 = digits_5(:,1:1200);
digits_6 = digits_6(:,1:1200);
digits_7 = digits_7(:,1:1200);
digits_8 = digits_8(:,1:1200);
digits_9 = digits_9(:,1:1200);
all_digits = [digits_0 digits_1 digits_2 digits_3 digits_4 digits_5 digits_6 digits_7 digits_8 digits_9];