clear all;
% weights goes from the least value of zero to 14; assuming sigma is 0.5
% and alpha =12, chosen randomly

%% create many realization of this distribution
%% log normal dist.
for i=1:100000
w(i)=exp(-0.0507+(0.3527*randn(1)));
end

load ('W_PN_KC.mat')
P_w= hist(w,[0:0.1:max(W)+0.1])/100000;
P_w=P_w+1e-6;

step=(max(W)/size(P_w,2));
% 
% W=[1e-6:step:max(w)];  % weights from 0.5 to 3.5 following turner et.al distrbution

% constrained that the min

%Mustep=(max(w)-min(w))/7;


kinit=7; 

%initM=[min(w)+Mustep:Mustep:max(w)];


initsigma=7;



%% p(w)= P(W|n)*P(n)

for claw=2:20
    
    initMu=log(kinit/(claw) );

    P_n(claw-1)= (1/(sqrt(2*pi)*1))*exp(-(( (claw)-6)^2)/(2*(1^2)));
    PW_given_n(claw-1,:)=(1/(sqrt(2*pi)* (initsigma) ))*(1./W).*exp(- ((log(W)-(initMu)).^2 )/(2* ((initsigma) ^2)) );

end

Pn=repmat(P_n',1,size(PW_given_n,2));

%start of the training

PW_tilda=sum(PW_given_n.*Pn);



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

        for claw=2:20
             
             Mu=log(k/(claw) );
            
%              term1= (-1/(Sigma^2))* exp(-(log(wi)-mu)^2/(2*(Sigma^2))) * ( 0.5/ ((mu-log(k/(claw)))^0.5) ) *(-1/k);
            
%              term2= (1/(Sigma^4))*(exp(-(log(wi)-mu)^2/(2*(Sigma^2))))* ((log(wi)-mu)^2)* ( 0.5/ ((mu-log(k/(claw)))^0.5) ) *(-1/k);
            
            
            dpW_given_n_dk(claw-1)=(1/(sigma*wi*sqrt(2*pi)))* (exp(-(log(wi)-Mu)^2/(2*(sigma^2)))) * (-2*(log(wi)-Mu)) *(-1/k);
            %(1/(wi*sqrt(2*pi)*Sigma))*(1/(2*(Sigma^2)))* (exp(-(log(wi)-log(k/claw))^2/(2*(Sigma^2))))*  (-2*(log(wi)-log(k/claw))) *(claw/k);   

        end
        
        dp_dk= sum(P_n.*dpW_given_n_dk);
        
        if (~isreal(dp_dk))
            see=1;
        end
        
        temp= -dp_dk*(P_w(weight)/PW_tilda(weight)) ;
        
        if (isnan(temp) || isinf(abs(temp)))
            temp=0;
        end
        
        GradJ_k= GradJ_k+temp;
        
        if (isinf(abs(GradJ_k)))
             br=0;
         end
    end
    
    trials=trials+1;

    GradJ_k_vector(trials)= GradJ_k;
    
    k= k - ((0.00001)*  (GradJ_k));
    kvector(trials)=k;
    
    %% optimize over sigma
%     
     for weight=1:size(W,2)
         
% 
         wi=W(weight);
% 
         for claw=2:20
              
              Mu=log(k/(claw) );

              term1= (-1/(sigma^2))* exp(-(log(wi)-Mu)^2/(2*(sigma^2))) ;
            
              term2= (1/(sigma^4))*(exp(-(log(wi)-Mu)^2/(2*(sigma^2))))* ((log(wi)-Mu)^2);
%             
%             
             dpW_given_n_dSigma(claw-1)=(1/(wi*sqrt(2*pi)))*(term1 + term2);   
% 
         end
% 
         dp_dSigma= sum(P_n.*dpW_given_n_dSigma);
% 
         tempS=-dp_dSigma*(P_w(weight)/PW_tilda(weight));
         
         if (isnan(tempS) || isinf(abs(tempS)))
            tempS=0;
        end
         
         GradJ_sigma= GradJ_sigma+tempS;
         
         if (isinf(abs(GradJ_sigma)))
             br=0;
         end
% 
    end
%     
     sigma=sigma- ((0.00001)* (GradJ_sigma));
    
     sigvector(trials)=sigma;

    %% optimize over sigma
%     
   %%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% calculate new PW_tilda
    
    for claw=2:20
        
     Mu=log(k/(claw) );

    P_n(claw-1)= (1/(sqrt(2*pi)*1))*exp(-(( (claw)-6)^2)/(2*(1^2)));
    PW_given_n(claw-1,:)=(1/(sqrt(2*pi)* (sigma) ))*(1./W).*exp(- ((log(W)-(Mu)).^2 )/(2* ((sigma) ^2)) );
    
    PW_given_n(claw-1,:)=(PW_given_n(claw-1,:))./sum(PW_given_n(claw-1,:),2);  
    end

    Pn=repmat(P_n',1,size(PW_given_n,2));

    %% calculate new PW_tilda


    PW_tilda=sum(PW_given_n.*Pn);

    
%       PW_tilda=(PW_tilda)./sum(PW_tilda,2);
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

%save ('W_PN_KC.mat','W');
save ('PW_given_N.mat', 'PW_given_n');

save('P_n.mat','P_n');
