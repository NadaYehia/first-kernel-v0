clear all;
% weights goes from the least value of zero to 14; assuming sigma is 0.5
% and alpha =12, chosen randomly

%% create many realization of this distribution
%% log normal dist.
for i=1:1000000
w(i)=exp(-0.0507+(0.3527*randn(1)));
end

P_w= hist(w,[0:0.1:max(w)]);
P_w=P_w+1e-6;

step=(max(w)/size(P_w,2));

W=[1e-6:step:max(w)];  % weights from 0.5 to 3.5 following turner et.al distrbution



kinit=30; 

%initM=[min(w)+Mustep:Mustep:max(w)];


initsigma=2;



%% p(w)= P(W|n)*P(n)
%% theta 1 is the minimum value of the spiking threshold, 
%% theta2 is the maximum value of the spiking threshold
%% create a distribution of the theta values, approximated to be gaussian, with relative coeffecient of variation 
%% sigma/Mu= 5.6/21.5. 
%% and create the P(claws) as gaussian distrbution from 2-12 claws.

claw=[2:15];

Xtheta= randn(1,10000)+(21.5/5.6);

P_n= (1/(sqrt(2*pi)))*exp(-(( (claw)-6).^2)./(2*(1^2)) );

%% by seeing the histogram of this dist. it is gaussian


 theta1= min(Xtheta);
 theta2=max(Xtheta);

j=1;
i=1;

%create the P(w|n,theta) where the median of these lognormals is prop. to (theta/n).. 

for claw=2:15
    for theta= 0.0003:0.0003:1
    
    initMu=log((kinit*theta)/claw );
    %% sigma of the theta dist. is 1..
    Ptheta(j)=(1/sqrt(2*pi*(0.05^2)) )*exp(- ((theta-(0.05*21.5/5.6)).^2) /(2* ((0.05) ^2)));
    PW_given_theta_and_n(i,j,:)=(1/(sqrt(2*pi)* (initsigma) ))*(1./W).*exp(- ((log(W)-(initMu)).^2 )/(2* ((initsigma) ^2)) );
    j=j+1;
    
    end
    i=i+1;
    j=1;
end

P_theta=repmat(Ptheta,size(PW_given_theta_and_n,1),1,size(PW_given_theta_and_n,3));
Pn=repmat(P_n',1,size(PW_given_theta_and_n,2),size(PW_given_theta_and_n,3));

%start of the training

PW_tilda=(PW_given_theta_and_n.*Pn.*P_theta);
PW_tilda= sum(sum(PW_tilda,1),2);
PW_tilda=permute(PW_tilda,[3,2,1]);

PW_tilda=PW_tilda';

 PW_tilda=(PW_tilda)./sum(PW_tilda,2);
 P_w=P_w./sum(P_w,2);

 figure,plot(W, PW_tilda);
 hold on, plot(W,P_w)

%% gradient descent optimization on Mu

k=kinit;
trials=0;
sigma=initsigma;
error=10;



while(abs(error)>0.1)
    
    GradJ_sigma=0;
    GradJ_k=0;
    
    for weight=1:size(W,2)

        wi=W(weight);
        ind=1;
       for claw=2:15
       
           for theta=0.0003:0.0003:1
             
             Mu=log((k*(theta))/ claw);
                   
            
            dpW_given_theta_and_n_dk(claw-1,ind)=(1/(sigma*wi*sqrt(2*pi)))* (exp(-(log(wi)-Mu)^2/(2*(sigma^2)))) * (-2*(log(wi)-Mu)) *(-1/k);
            ind=ind+1;

        
           end
           ind=1;
       end
        
       POfn=repmat(P_n',1,size(dpW_given_theta_and_n_dk,2));
       POfTheta=repmat(Ptheta,size(dpW_given_theta_and_n_dk,1),1);
        
       
        dp_dk= sum(sum(POfn.*POfTheta.*dpW_given_theta_and_n_dk));
        
       
        
        temp= -dp_dk*(P_w(weight)/PW_tilda(weight)) ;
        
        if (isnan(temp) || isinf(abs(temp)))
            temp=0;
        end
        
        GradJ_k= GradJ_k+temp;
        
       
    end
    
    trials=trials+1;

    GradJ_k_vector(trials)= GradJ_k;
    
    k= k - ((1e-6)*  (GradJ_k));
    kvector(trials)=k;
    
    %% optimize over sigma
%     
     for weight=1:size(W,2)
         
% 
         wi=W(weight);
         ind2=1;
       for claw=2:15 
           
         for theta=0.0003:0.0003:1
              
              Mu=log((k*(theta))/claw );

              term1= (-1/(sigma^2))* exp(-(log(wi)-Mu)^2/(2*(sigma^2))) ;
            
              term2= (1/(sigma^4))*(exp(-(log(wi)-Mu)^2/(2*(sigma^2))))* ((log(wi)-Mu)^2);
             
             
             dpW_given_theta_and_n_dSigma(claw-1,ind2)=(1/(wi*sqrt(2*pi)))*(term1 + term2);  
             ind2=ind2+1;
 
         end
         ind2=1;
       end
       
       POfn=repmat(P_n',1,size(dpW_given_theta_and_n_dk,2));
       POfTheta=repmat(Ptheta,size(dpW_given_theta_and_n_dk,1),1);
         
         dp_dSigma= sum(sum(POfn.*POfTheta.*dpW_given_theta_and_n_dSigma)); 
         
         tempS=-dp_dSigma*(P_w(weight)/PW_tilda(weight));
         
         if (isnan(tempS) || isinf(abs(tempS)))
            tempS=0;
        end
         
         GradJ_sigma= GradJ_sigma+tempS;
         
        
% 
    end
%     
     sigma=sigma- ((1e-6)* (GradJ_sigma));
    
     sigvector(trials)=sigma;

    %% optimize over sigma
%     
   %%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% calculate new PW_tilda
    kind=1; 

   for claw=2:15
    for theta=0.0003:0.0003:1
       
     Mu=log(k*(theta)/claw );

    PW_given_theta_and_n(claw-1,kind,:)=(1/(sqrt(2*pi)* (sigma) ))*(1./W).*exp(- ((log(W)-(Mu)).^2 )/(2* ((sigma) ^2)) );
    
    kind=kind+1;
    
    end
    kind=1;
   end


    %% calculate new PW_tilda

    
    P_theta=repmat(Ptheta,size(PW_given_theta_and_n,1),1,size(PW_given_theta_and_n,3));
    Pn=repmat(P_n',1,size(PW_given_theta_and_n,2),size(PW_given_theta_and_n,3));

%start of the training

    PW_tilda=(PW_given_theta_and_n.*Pn.*P_theta);
    PW_tilda= sum(sum(PW_tilda,1),2);
    PW_tilda=permute(PW_tilda,[3,2,1]);
    PW_tilda=PW_tilda';

    
    PW_tilda=(PW_tilda)./sum(PW_tilda,2);
    P_w=P_w./sum(P_w,2);
    temp=P_w.* log(P_w./PW_tilda);
    
    temp(isinf(temp))=0;
    error= sum(temp);
    j(trials) = error;
    
end


%PW_given_n{claw}=(1/(sqrt(2*pi)*sigma))*exp(- ((W-(alpha/n))^2 )/(2*(sigma^2)) );


figure,plot(W,P_w)
hold on,plot(W,PW_tilda)

% figure
% for i=1:10
% 
% hold on, plot(W,PW_given_n(i,:));
% end

save ('W_PN_KC.mat','W');
save ('PW_given_theta_and_n.mat', 'PW_given_theta_and_n');

save('Ptheta.mat','Ptheta');
save('P_n.mat','P_n');
