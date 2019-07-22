clear all; 

%load hallem .mat
load('hallem_olsen.mat');
load('kazama.mat'); %84 PNs *37 odors

load('PW_given_N.mat');
load('PW_given_theta_and_n.mat');
load('W_PN_KC.mat');
% load('classAction1.mat');
load ('P_n.mat');

ll=20;
numtrainingSamples=30; %% artificial plus linearly dependent responses 
modss=1;
ods=1;
mulOd=100;
lrs=6;
Crange=5;


p_ra=zeros(ods,lrs,ll,numtrainingSamples,1);
p_raEq=zeros(ods,lrs,ll, numtrainingSamples,1);

test_p_ra=zeros(Crange,ll,lrs);
test_p_raEq=zeros(Crange,ll,lrs);
test_p_raH=zeros(Crange,ll,lrs);
test_p_ra_Fixedtheta= zeros(Crange,ll,lrs);
test_p_raEq_FixedTheta_Nonly= zeros(Crange,ll,lrs);
test_p_raH_FixedTheta=zeros(Crange,ll,lrs);
test_p_raPNs=zeros(Crange,ll,lrs);
% 

angDistSimple=zeros(lrs,ll);
angDistCompensated=zeros(lrs,ll);
angDistHomogenous=zeros(lrs,ll);
angDistSimple_Ftheta=zeros(lrs,ll);
angDistCompensated_Ftheta=zeros(lrs,ll);
angDistHomogenous_Ftheta=zeros(lrs,ll);
%        
% angDistSimple=zeros(mulOd,mulOd,ll);
% angDistCompensated=zeros(mulOd,mulOd,ll);
% angDistHomogenous=zeros(mulOd,mulOd,ll);
% angDistSimple_Ftheta=zeros(mulOd,mulOd,ll);
% angDistCompensated_Ftheta=zeros(mulOd,mulOd,ll);
% angDistHomogenous_Ftheta=zeros(mulOd,mulOd,ll);
%            
odors=mulOd;
numTrials = 60;
PN = hallem_olsen(1:110,:);
% PN=(PN-mean(PN,2))./repmat(std(PN,0,2),1,odors);

PNs=zeros(24,odors);
PNtrials = zeros(24, odors, numTrials);


% Generate noisy PN activity
        % in the "trials" dimension, trial = 1 is without any noise,
        % while indices above 1 are with added noise. Thus trial = 1 is
        % the "average" trial

        % if a scalar number, this parameter specifies the coefficient of
        % variation of PN activity


        k=0; 
        %% create artificial odors, n odors

            for Pn=1:24
                [prob,bins]=hist(PN(Pn,:));
                prob=prob/sum(prob);

                PNs(Pn,k+1:k+odors)=randsample(bins,odors,'true',prob);

            end

        PNtrials(:,:,1) =PNs;

        for t = 1:numTrials-1
                            PNtrials(:,:,t+1) = PNs + ...
                                getPNStdevBhandawat(PNs) .* ...
                                randn(24, odors);
        end


         PNtrials=(PNtrials - min(PNtrials(:)))/(max(PNtrials(:))- min(PNtrials(:)));
%          PNtrials=PNtrials-mean( mean(PNtrials,3) ,2);

PNtrials=reshape(PNtrials,24,odors*numTrials);

%% create a 2D array for each model
%% no of od casting X 2 (performance and dBI)
%% each point is the average performance and DBI across
%% all random trials


% Perf_Dbi_simpleFixedTheta= zeros(modss*K,2);
% Perf_Dbi_Simple= zeros(modss*K,2);
% Perf_Dbi_Compensated= zeros(modss*K,2);
% Perf_Dbi_CompensatedFixedTheta= zeros(modss*K,2);
% Perf_Dbi_Homogenous= zeros(modss*K,2);
% Perf_Dbi_HomogenousFixedTheta= zeros(modss*K,2);

%mods=6;
     C_SoftMax=0.0001;
     
     for mods=1:modss
        odors= mulOd*(mods/modss);
     
       for randomTrials=1:ll
           
       
                classAction1=randsample([1:odors],round(odors/2),'false');
           
                alpha=0.00001;
                
                theta_threshold= 0.0+(1.8);
                

                InhibitionGain= 0.0+ (0.275);

                n =2000; % number of neurons in the hidden layer

                m=24;  %number of dimensions in the input data

                clawsNo=(normrnd(6,1,[1 n])); %select number of claws randomly
                
                HomogClaws= ones([1,n])*6; 

                PNsperKC = floor(clawsNo.*ones(1,n));

%                 PNsperKC(PNsperKC>10)=10;
                
                HomogPNsperKC= HomogClaws;

                 %% randomly assign thresholds for KCs
                 ThetaMu=0.22;
                 
                theta=abs(normrnd(ThetaMu,ThetaMu*(5.6/21.5),[1 n])); %% avoid negative values of theta
                theta(theta>1)=1;
                
                for i=1:n
                    PnToKc{i} = randsample(m, PNsperKC(i), true);
                    HomogPnToKc{i}= randsample(m, HomogPNsperKC(i), true);
                end % random initilaization of the weights

                %initialize the weights matrix between KC and PN

                thisW = zeros(m, n);
                thisW_equalizedModel=zeros(m,n);
                thisW_HomogModel=zeros(m,n);
                thisW_equalizedModel_Ftheta=zeros(m,n);
                
              

                for k=1:n

                    for j=1:length(PnToKc{k})

                      whichPN = PnToKc{k}(j);
                        % pick random weight from a log normal distribution that
                        % roughtly fits the Turner distribution
        
                       thisWeight = exp(-0.0507+0.3527*randn(1));

                      %% sample the weights from the new fitted weights in the other script (modelling KC_PnWeights.m)
                      
                       ThetaInd= round(((theta(k)-0.0003)/0.0003)+1);
                       ThetaInd(ThetaInd==0)=1;
                        
                      
                      
                       this_KCWeights= PW_given_theta_and_n(length(PnToKc{k}),ThetaInd,:);
                       thisWeight_equalizedModel= randsample(W,1,'true', this_KCWeights);

                       this_KCWeights_Nonly= PW_given_n(length(PnToKc{k}),:);
                       thisWeight_equalizedModel_Ftheta= randsample(W,1,'true', this_KCWeights_Nonly);

                       
                       % have to keep track of all weights in this way rather than w(w>0)
                       % because some weights are doubled ie a KC can be connected to the
                       % same PN channel more than once
                      thisW_equalizedModel_Ftheta(whichPN,k)= thisW_equalizedModel_Ftheta(whichPN,k) +thisWeight_equalizedModel_Ftheta; 
                      thisW(whichPN, k) = thisW(whichPN, k) + thisWeight;
                      thisW_equalizedModel(whichPN,k)= thisW_equalizedModel(whichPN,k)+thisWeight_equalizedModel;


                    end
                end
                
                
                for k=1:n

                    for j=1:length(HomogPnToKc{k})

                      
                      whichPN_homog= HomogPnToKc{k}(j);
        
                      thisWeightHomo=1; %% homogenous equal unity weights connecting KCs to PNs.

                      %% sample the weights from the new fitted weights in the other script (modelling KC_PnWeights.m)

                     
                      thisW_HomogModel(whichPN_homog,k)= thisWeightHomo+ thisW_HomogModel(whichPN_homog,k); 


                    end
                end
                
                

                %% find the threshold to fix the coding levels the same for this odor number,random
                %% trial and across all training samples
                thetaS_Ftheta=theta_threshold;
                thetaH_Ftheta=theta_threshold;
                
                thetaS=theta;
                thetaH=theta;
                
                tempMuS=ThetaMu;
                tempMuH=ThetaMu;
                
                mSimp=0.09;
                
                mComp=0.1;
                mHomog=0.4;
                mHomog_Ftheta=0.4;
                mSimp_Ftheta=0.4;
                mComp_Ftheta=0.4;
                
                step=1.0;

               ActivationsDummy=zeros(n,odors*numtrainingSamples);
               ActivationsEqualizeddummy=zeros(n,odors*numtrainingSamples);
               ActivationsHomogenousdummy=zeros(n,odors*numtrainingSamples);
               
               ActivationsDummy_Ftheta=zeros(n,odors*numtrainingSamples);
               ActivationsEqualizeddummy_Ftheta=zeros(n,odors*numtrainingSamples);
               ActivationsHomogenousdummy_Ftheta=zeros(n,odors*numtrainingSamples);


               Ydummy=zeros(n,odors*numtrainingSamples);
               YEqualizeddummy=zeros(n,odors*numtrainingSamples);
               YHomogdummy=zeros(n,odors*numtrainingSamples);
               
               Ydummy_Ftheta=zeros(n,odors*numtrainingSamples);
               YEqualizeddummy_Ftheta=zeros(n,odors*numtrainingSamples);
               YHomogdummy_Ftheta=zeros(n,odors*numtrainingSamples);
               
               


                 while (abs(mSimp-mComp)>0.001)

                     

                     for trial = 1:(odors*numtrainingSamples) 

                         if (~mod(trial,odors)) 
                             
                        ActivationsDummy(:,trial) = thisW'*PNtrials(:,trial+ ( (floor(trial/odors)-1) *(mulOd-odors)) );
                        Ydummy(:,trial)=(( ActivationsDummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsDummy(:,trial),1),n,1)-thetaS')>0 ).*( ActivationsDummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsDummy(:,trial),1),n,1)-thetaS');
                        codingLevelDummy(trial)=  (sum(Ydummy(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + ( (floor(trial/odors)-1) *(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);

                         
                         
                         else 
                             
                        ActivationsDummy(:,trial) = thisW'*PNtrials(:,trial+ (floor(trial/odors)*(mulOd-odors)) );
                        Ydummy(:,trial)=(( ActivationsDummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsDummy(:,trial),1),n,1)-thetaS')>0 ).*( ActivationsDummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsDummy(:,trial),1),n,1)-thetaS');
                        codingLevelDummy(trial)=  (sum(Ydummy(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);

                         end
                     end
                    mSimp=mean(codingLevelDummy);
                    mComp=mean(codingLevelEqualizedDummy);
                    tempMuS= tempMuS+(mSimp-mComp);
                    
                    thetaS=abs(normrnd(tempMuS,tempMuS*(5.6/21.5),[1 n])); %% avoid negative values of theta
                    thetaS(thetaS>20)=20;
                    
                 end
                    if (tempMuS<0)
                        stop_S=1;
                    end


      
                while (abs(mComp-mHomog)>0.001)

                     

                     for trial = 1:(odors*numtrainingSamples) 
                         
                        if (~mod(trial,odors) )
                            
                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)) );
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsHomogenousdummy(:,trial) = thisW_HomogModel'*PNtrials(:,trial + ( ( (floor(trial/odors)-1) *(mulOd-odors))) );
                        YHomogdummy(:,trial)=(( ActivationsHomogenousdummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomogenousdummy(:,trial),1),n,1)-thetaH')>0 ).*( ActivationsHomogenousdummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomogenousdummy(:,trial),1),n,1)-thetaH');
                        codingLevelHomoDummy(trial)=  (sum(YHomogdummy(:,trial)>0,1)/n);

                         
                            
                            
                        else

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)) );
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsHomogenousdummy(:,trial) = thisW_HomogModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                        YHomogdummy(:,trial)=(( ActivationsHomogenousdummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomogenousdummy(:,trial),1),n,1)-thetaH')>0 ).*( ActivationsHomogenousdummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomogenousdummy(:,trial),1),n,1)-thetaH');
                        codingLevelHomoDummy(trial)=  (sum(YHomogdummy(:,trial)>0,1)/n);

                        end
                    end
                    mComp=mean(codingLevelEqualizedDummy);
                    mHomog=mean(codingLevelHomoDummy);
                   
                    tempMuH=tempMuH+ (mHomog-mComp);
                   
                    thetaH=abs(normrnd(tempMuH,tempMuH*(5.6/21.5),[1 n])); %% avoid negative values of theta
                    thetaH(thetaH>20)=20;
                    
                    if(tempMuH<0)
                    flag_H=1;
                    end
                    
                end  
                
                
                 
                
                %% fix the coding level of the other models with fixed theta to be
                %% the same as the compensated model with variable theta.
                
                  while (abs(mSimp_Ftheta-mComp)>0.001)

                     

                     for trial = 1:(odors*numtrainingSamples) 
                         
                         if (~mod(trial,odors))
                             
                        ActivationsDummy_Ftheta(:,trial) = thisW'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)) );
                        Ydummy_Ftheta(:,trial)=(( ActivationsDummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsDummy_Ftheta(:,trial),1),n,1)-thetaS_Ftheta)>0 ).*( ActivationsDummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsDummy_Ftheta(:,trial),1),n,1)-thetaS_Ftheta);
                        codingLevelDummy_Ftheta(trial)=  (sum(Ydummy_Ftheta(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + ( (floor(trial/odors)-1) *(mulOd-odors)) );
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);
                             
                             
                         else
                        
                        ActivationsDummy_Ftheta(:,trial) = thisW'*PNtrials(:,trial +(floor(trial/odors)*(mulOd-odors)) );
                        Ydummy_Ftheta(:,trial)=(( ActivationsDummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsDummy_Ftheta(:,trial),1),n,1)-thetaS_Ftheta)>0 ).*( ActivationsDummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsDummy_Ftheta(:,trial),1),n,1)-thetaS_Ftheta);
                        codingLevelDummy_Ftheta(trial)=  (sum(Ydummy_Ftheta(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)) );
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);
                         end
                         
                    end
                    mSimp_Ftheta=mean(codingLevelDummy_Ftheta);
                    mComp=mean(codingLevelEqualizedDummy);
                    thetaS_Ftheta=thetaS_Ftheta+(mSimp_Ftheta-mComp);
                    
                  end

                    if (thetaS_Ftheta<0)
                        stop_SF=1;
                    end
                 
                  while (abs(mComp_Ftheta-mComp)>0.001)

                     

                     for trial = 1:(odors*numtrainingSamples) 
                         
                         if (~mod(trial,odors))
                             
                        ActivationsEqualizeddummy_Ftheta(:,trial) = thisW_equalizedModel_Ftheta'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)) );
                        YEqualizeddummy_Ftheta(:,trial)=(( ActivationsEqualizeddummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy_Ftheta(:,trial),1),n,1)-theta_threshold)>0 ).*( ActivationsEqualizeddummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy_Ftheta(:,trial),1),n,1)-theta_threshold);
                        codingLevelEqualizedDummy_Ftheta(trial)=  (sum(YEqualizeddummy_Ftheta(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);  
                             
                             
                             
                         else

                        ActivationsEqualizeddummy_Ftheta(:,trial) = thisW_equalizedModel_Ftheta'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                        YEqualizeddummy_Ftheta(:,trial)=(( ActivationsEqualizeddummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy_Ftheta(:,trial),1),n,1)-theta_threshold)>0 ).*( ActivationsEqualizeddummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy_Ftheta(:,trial),1),n,1)-theta_threshold);
                        codingLevelEqualizedDummy_Ftheta(trial)=  (sum(YEqualizeddummy_Ftheta(:,trial)>0,1)/n);

                        %% same but for the equalized model

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n);
                         end
                         
                    end
                    mComp_Ftheta=mean(codingLevelEqualizedDummy_Ftheta);
                    mComp=mean(codingLevelEqualizedDummy);
                    theta_threshold=theta_threshold+(mComp_Ftheta-mComp);
                    
                 end

                  if (theta_threshold<0)
                        stop_CF=1;
                  end

      
                while (abs(mComp-mHomog_Ftheta)>0.02)

                     

                     for trial = 1:(odors*numtrainingSamples) 
                         
                         if (~mod(trial,odors))
                         
                         ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n); 
                        %% same but for the equalized model

                        ActivationsHomogenousdummy_Ftheta(:,trial) = thisW_HomogModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                        YHomogdummy_Ftheta(:,trial)=(( ActivationsHomogenousdummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomogenousdummy_Ftheta(:,trial),1),n,1)-thetaH_Ftheta)>0 ).*( ActivationsHomogenousdummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomogenousdummy_Ftheta(:,trial),1),n,1)-thetaH_Ftheta);
                        codingLevelHomoDummy_Ftheta(trial)=  (sum(YHomogdummy_Ftheta(:,trial)>0,1)/n);
   
                         else

                        ActivationsEqualizeddummy(:,trial) = thisW_equalizedModel'*PNtrials(:,trial +( (floor(trial/odors)) *(mulOd-odors)));
                        YEqualizeddummy(:,trial)=(( ActivationsEqualizeddummy(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualizeddummy(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualizeddummy(:,trial),1),n,1)-theta');
                        codingLevelEqualizedDummy(trial)=  (sum(YEqualizeddummy(:,trial)>0,1)/n); 
                        %% same but for the equalized model

                        ActivationsHomogenousdummy_Ftheta(:,trial) = thisW_HomogModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                        YHomogdummy_Ftheta(:,trial)=(( ActivationsHomogenousdummy_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomogenousdummy_Ftheta(:,trial),1),n,1)-thetaH_Ftheta)>0 ).*( ActivationsHomogenousdummy_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomogenousdummy_Ftheta(:,trial),1),n,1)-thetaH_Ftheta);
                        codingLevelHomoDummy_Ftheta(trial)=  (sum(YHomogdummy_Ftheta(:,trial)>0,1)/n);

                         end
                    end
                    mComp=mean(codingLevelEqualizedDummy);
                    mHomog_Ftheta=mean(codingLevelHomoDummy_Ftheta);
                    
                   
                    
                   
                 thetaH_Ftheta=thetaH_Ftheta+ (mHomog_Ftheta-mComp);
                    
                    
                 end  
                    if (thetaH_Ftheta<0)
                        stop_HF=1;
                    end


               Activations=zeros(n,odors*numTrials);
               ActivationsEqualized=zeros(n,odors*numTrials);
               ActivationsHomog=zeros(n,odors*numTrials);

               Activations_Ftheta=zeros(n,odors*numTrials);
               ActivationsEqualized_Ftheta=zeros(n,odors*numTrials);
               ActivationsHomog_Ftheta=zeros(n,odors*numTrials);


               Y=zeros(n,odors*numTrials);
               YEqualized=zeros(n,odors*numTrials);
               YHomog=zeros(n,odors*numTrials);
               
               
               Y_Ftheta=zeros(n,odors*numTrials);
               YEqualized_Ftheta=zeros(n,odors*numTrials);
               YHomog_Ftheta=zeros(n,odors*numTrials);
               

    

               for trial = 1:(odors*numTrials)
                   
                   if (~mod(trial,odors))
                       
                   ActivationsHomog_Ftheta(:,trial) = thisW_HomogModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)) );
                   YHomog_Ftheta(:,trial)=(( ActivationsHomog_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomog_Ftheta(:,trial),1),n,1)-thetaH_Ftheta)>0 ).*( ActivationsHomog_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomog_Ftheta(:,trial),1),n,1)-thetaH_Ftheta);
                   
                   Activations_Ftheta(:,trial) = thisW'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                    Y_Ftheta(:,trial)=(( Activations_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(Activations_Ftheta(:,trial),1),n,1)-thetaS_Ftheta)>0 ).*( Activations_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(Activations_Ftheta(:,trial),1),n,1)-thetaS_Ftheta);
       
                    ActivationsEqualized_Ftheta(:,trial) = thisW_equalizedModel_Ftheta'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                    YEqualized_Ftheta(:,trial)=(( ActivationsEqualized_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualized_Ftheta(:,trial),1),n,1)-theta_threshold)>0 ).*( ActivationsEqualized_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualized_Ftheta(:,trial),1),n,1)-theta_threshold);
                    
                    
                    
                    
                   ActivationsHomog(:,trial) = thisW_HomogModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                   YHomog(:,trial)=(( ActivationsHomog(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomog(:,trial),1),n,1)-thetaH')>0 ).*( ActivationsHomog(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomog(:,trial),1),n,1)-thetaH');
                   
                   Activations(:,trial) = thisW'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                    Y(:,trial)=(( Activations(:,trial)-(InhibitionGain)/(n)*repmat(sum(Activations(:,trial),1),n,1)-thetaS')>0 ).*( Activations(:,trial)-InhibitionGain/(n)*repmat(sum(Activations(:,trial),1),n,1)-thetaS');
       
                   

                    ActivationsEqualized(:,trial) = thisW_equalizedModel'*PNtrials(:,trial +( (floor(trial/odors)-1) *(mulOd-odors)));
                    YEqualized(:,trial)=(( ActivationsEqualized(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualized(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualized(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualized(:,trial),1),n,1)-theta');
        
                   
                   else
                       
                   ActivationsHomog_Ftheta(:,trial) = thisW_HomogModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)) );
                   YHomog_Ftheta(:,trial)=(( ActivationsHomog_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomog_Ftheta(:,trial),1),n,1)-thetaH_Ftheta)>0 ).*( ActivationsHomog_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomog_Ftheta(:,trial),1),n,1)-thetaH_Ftheta);
                   
                   Activations_Ftheta(:,trial) = thisW'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                    Y_Ftheta(:,trial)=(( Activations_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(Activations_Ftheta(:,trial),1),n,1)-thetaS_Ftheta)>0 ).*( Activations_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(Activations_Ftheta(:,trial),1),n,1)-thetaS_Ftheta);
       
                    ActivationsEqualized_Ftheta(:,trial) = thisW_equalizedModel_Ftheta'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                    YEqualized_Ftheta(:,trial)=(( ActivationsEqualized_Ftheta(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualized_Ftheta(:,trial),1),n,1)-theta_threshold)>0 ).*( ActivationsEqualized_Ftheta(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualized_Ftheta(:,trial),1),n,1)-theta_threshold);
                    
                    
                    
                    
                   ActivationsHomog(:,trial) = thisW_HomogModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                   YHomog(:,trial)=(( ActivationsHomog(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomog(:,trial),1),n,1)-thetaH')>0 ).*( ActivationsHomog(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomog(:,trial),1),n,1)-thetaH');
                   
                   Activations(:,trial) = thisW'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                    Y(:,trial)=(( Activations(:,trial)-(InhibitionGain)/(n)*repmat(sum(Activations(:,trial),1),n,1)-thetaS')>0 ).*( Activations(:,trial)-InhibitionGain/(n)*repmat(sum(Activations(:,trial),1),n,1)-thetaS');
       
                     %rho(:,:,trial)=corrcoef(Y(:,:,trial));
                    %% same but for the equalized model

                    ActivationsEqualized(:,trial) = thisW_equalizedModel'*PNtrials(:,trial + (floor(trial/odors)*(mulOd-odors)));
                    YEqualized(:,trial)=(( ActivationsEqualized(:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualized(:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualized(:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualized(:,trial),1),n,1)-theta');
                   
                   end
     
          
               %[h,p]=ttest2(avgCov_simple,avgCov_equalized);
                
        for l_r=1:lrs  
               
                WopAllOdours=1*rand(n,2);
                WopAllOdoursEqualized= WopAllOdours;
                WopAllOdoursHomog=WopAllOdours;
                
                WopAllOdours_Ftheta=WopAllOdours;
                WopAllOdoursEqualized_Ftheta= WopAllOdours;
                WopAllOdoursHomog_Ftheta=WopAllOdours;
                
                WopFromPNs= 1*rand(m,2);
                
                dim_S(mods,randomTrials)= dimInputCurrent(Calc_C(Y));
                dim_SF(mods,randomTrials)=dimInputCurrent(Calc_C(Y_Ftheta));
                
                dim_C(mods,randomTrials)=dimInputCurrent(Calc_C(YEqualized));
                dim_CF(mods,randomTrials)=dimInputCurrent(Calc_C(YEqualized_Ftheta));
                
                dim_H(mods,randomTrials)=dimInputCurrent(Calc_C(YHomog));
                dim_HF(mods,randomTrials)=dimInputCurrent(Calc_C(YHomog_Ftheta));
                
                dim_PNs(mods,randomTrials)= dimInputCurrent(Calc_C(PNtrials));

                %% take the covariance of the odor responses with each other

                alpha=0.0001* (10^((l_r)));
                
                c=1;
                ceq=1;
                ch=1;

%               PNtrials
                
                %PNtrialstemp=reshape(PNtrials,m,odors,numTrials);
                %PNtrialstr=PNtrialstemp(:,:,1:numtrainingSamples);
              
                YHomogtemp=reshape(YHomog,n,odors,numTrials);
                YHomog_Fthetatemp=reshape(YHomog_Ftheta,n,odors,numTrials);
                Ytemp= reshape(Y,n,odors,numTrials);
                Y_Fthetatemp= reshape(Y_Ftheta,n,odors,numTrials);
                YEqualizedtemp=reshape(YEqualized,n,odors,numTrials);
                YEqualized_Fthetatemp= reshape(YEqualized_Ftheta,n,odors,numTrials);

                YHomogtr=YHomogtemp(:,:,1:numtrainingSamples);
                YHomog_Fthetatr=YHomog_Fthetatemp(:,:,1:numtrainingSamples);
                
                Ytr=Ytemp(:,:,1:numtrainingSamples);
                Y_Fthetatr=Y_Fthetatemp(:,:,1:numtrainingSamples);
                
                YEqualizedtr=YEqualizedtemp(:,:,1:numtrainingSamples);
                YEqualized_Fthetatr=YEqualized_Fthetatemp(:,:,1:numtrainingSamples);
                %% learning from a preceptron in the output layer
                
                
%                 for trials=1:numtrainingSamples

                 for odour=1:odors                    
                     
                        %% prob. of success in equalized model
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       if( ~ isempty(find(classAction1==odour)) )             
                             delta =  exp( -(alpha/mean(YHomogtr(:)))* sum(YHomogtr(:,odour,:),3) );
                             
                             deltaWH(:,ch)= WopAllOdoursHomog(:,2).*(delta-1);
                             ch=ch+1;
                             WopAllOdoursHomog(:,2)= WopAllOdoursHomog(:,2) .*delta;

                        else

                              delta = exp(- (alpha/mean(YHomogtr(:)))* sum(YHomogtr(:,odour,:),3) );
                              WopAllOdoursHomog(:,1)= WopAllOdoursHomog(:,1) .*delta;

                       end
                       
                       %%
                        if( ~ isempty(find(classAction1==odour)) )             
                             delta =  exp( -(alpha/mean(YHomog_Fthetatr(:)))* sum(YHomog_Fthetatr(:,odour,:),3) );
                        
                             WopAllOdoursHomog_Ftheta(:,2)= WopAllOdoursHomog_Ftheta(:,2) .*delta;

                        else

                              delta = exp(- (alpha/mean(YHomog_Fthetatr(:)))* sum(YHomog_Fthetatr(:,odour,:),3) );
                              WopAllOdoursHomog_Ftheta(:,1)= WopAllOdoursHomog_Ftheta(:,1) .*delta;

                       end
                       
                       %%
                       
                        
                        if( ~ isempty(find(classAction1==odour)) )              
                             delta =  exp( -(alpha/mean(Ytr(:)))* sum(Ytr(:,odour,:),3) );
                             
                             deltaW(:,c)= WopAllOdours(:,2).*(delta-1);
                             c=c+1;
                             WopAllOdours(:,2)= WopAllOdours(:,2) .*delta;

                        else

                              delta = exp(- (alpha/mean(Ytr(:)))* sum(Ytr(:,odour,:),3) );
                              WopAllOdours(:,1)= WopAllOdours(:,1) .*delta;

                        end
                        
                        %%
                        if( ~ isempty(find(classAction1==odour)) )              
                             delta =  exp( -(alpha/mean(Y_Fthetatr(:)))* sum(Y_Fthetatr(:,odour,:),3) );
                            
                             WopAllOdours_Ftheta(:,2)= WopAllOdours_Ftheta(:,2) .*delta;

                        else

                              delta = exp(- (alpha/mean(Y_Fthetatr(:)))* sum(Y_Fthetatr(:,odour,:),3) );
                              WopAllOdours_Ftheta(:,1)= WopAllOdours_Ftheta(:,1) .*delta;

                        end
                        
                        %%
                        
                        if( ~ isempty(find(classAction1==odour)) )              

                             delta = exp( -(alpha/mean(YEqualizedtr(:)))* sum((YEqualizedtr(:,odour,:)),3) );
                             
                             
                             WopAllOdoursEqualized(:,2)= WopAllOdoursEqualized(:,2).*delta;

                        else

                              delta =  exp(-(alpha /mean(YEqualizedtr(:)))* sum(YEqualizedtr(:,odour,:),3) );
                             WopAllOdoursEqualized(:,1)= WopAllOdoursEqualized(:,1) .*delta;

                        end

                        %%
                         if( ~ isempty(find(classAction1==odour)) )              

                             delta = exp( -(alpha/mean(YEqualized_Fthetatr(:)))* sum((YEqualized_Fthetatr(:,odour,:)),3) );
                             
                             WopAllOdoursEqualized_Ftheta(:,2)= WopAllOdoursEqualized_Ftheta(:,2).*delta;

                        else

                              delta =  exp(-(alpha /mean(YEqualized_Fthetatr(:)))* sum(YEqualized_Fthetatr(:,odour,:),3) );
                             WopAllOdoursEqualized_Ftheta(:,1)= WopAllOdoursEqualized_Ftheta(:,1) .*delta;

                         end
                         
%                          if( ~ isempty(find(classAction1==odour)) )              
% 
%                              delta = exp( -(alpha/mean(PNtrialstr(:)))* sum((PNtrialstr(:,odour,:)),3) );
%                              
%                              WopFromPNs(:,2)= WopFromPNs(:,2).*delta;
% 
%                         else
% 
%                               delta =  exp(-(alpha /mean(PNtrialstr(:)))* sum(PNtrialstr(:,odour,:),3) );
%                              WopFromPNs(:,1)= WopFromPNs(:,1) .*delta;
% 
%                          end
                         
                         
                         
                 end
                
                 %% perfromance as a function of the strictness of the decision making
                 %% this strictness is dictated by C in the soft-max function. 
                 %% so given the same fly, same task, and after learning measure the performance as f(c)
                 
             for c=1:Crange
                 
                C= C_SoftMax*(10^c);
                
                [acc,accEq,accH]=KernelTesting(C,WopAllOdours,WopAllOdoursEqualized,WopAllOdoursHomog,PNtrials,PnToKc,HomogPnToKc,theta,thetaS, thetaH,InhibitionGain,classAction1,numTrials,numtrainingSamples,Ytemp,YEqualizedtemp,YHomogtemp);
                
                test_p_raEq(c,randomTrials,l_r)=accEq;
                test_p_ra(c,randomTrials,l_r)=acc;
                test_p_raH(c,randomTrials,l_r)=accH;
                

              [acc1,accEq1,accH1]=KernelTesting_Ftheta(C,WopAllOdours_Ftheta,WopAllOdoursEqualized_Ftheta,WopAllOdoursHomog_Ftheta,PNtrials,PnToKc,HomogPnToKc,theta_threshold,thetaS_Ftheta, thetaH_Ftheta,InhibitionGain,classAction1,numTrials,numtrainingSamples,Y_Fthetatemp,YEqualized_Fthetatemp,YHomog_Fthetatemp);
                
                test_p_ra_Fixedtheta(c,randomTrials,l_r)=acc1;
                test_p_raEq_FixedTheta_Nonly(c,randomTrials,l_r)=accEq1;
                test_p_raH_FixedTheta(c,randomTrials,l_r)=accH1;   
                
%                 [accPns_ra]=KernelTesting_PNs (C,WopFromPNs,PNtrialstemp,classAction1,numTrials,numtrainingSamples);
%                 test_p_raPNs(mods,randomTrials)=accPns_ra;
%                 
                
              end
               

                

        
    
    
       end

%         end

    end
