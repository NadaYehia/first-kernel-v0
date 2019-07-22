function [p_ra,p_raEq,p_raH]=KernelTesting_Ftheta (C,Wop,WopEq,WopHom,PNs,PnToKc,HomogPnToKc,theta,thetaS,thetaH,InhibitionGain,classAction1,numTrials,numtrainingSamples,Y,YEqualized,YHomog)

load('PW_given_N.mat');
load('W_PN_KC.mat');
load ('P_n.mat');

numm=numTrials-numtrainingSamples;

n=2000;
% thisW = zeros(24, n);
% thisW_equalizedModel=zeros(24,n);
% thisW_HomogModel=zeros(24,n);
% 
%                 for k=1:n
% 
%                     for j=1:length(PnToKc{k})
% 
%                       whichPN = PnToKc{k}(j);
%                         % pick random weight from a log normal distribution that
%                         % roughtly fits the Turner distribution
%         
%                        thisWeight = exp(-0.0507+0.3527*randn(1));
%                        
% 
%                        %% sample the weights from the new fitted weights in the other script (modelling KC_PnWeights.m)
% 
%                        this_KCWeights= PW_given_n(length(PnToKc{k}),:);
%                        
%                        thisWeight_equalizedModel= randsample(W,1,'true', this_KCWeights);
% 
%                        % have to keep track of all weights in this way rather than w(w>0)
%                        % because some weights are doubled ie a KC can be connected to the
%                        % same PN channel more than once
%                       thisW(whichPN, k) = thisW(whichPN, k) + thisWeight;
%                       thisW_equalizedModel(whichPN,k)= thisW_equalizedModel(whichPN,k)+thisWeight_equalizedModel;
% 
% 
%                     end
%                 end
%                 
%                 
%                for k=1:n
% 
%                     for j=1:length(HomogPnToKc{k})
% 
%                       
%                       whichPN_homog= HomogPnToKc{k}(j);
%         
%                       thisWeightHomo=1; %% homogenous equal unity weights connecting KCs to PNs.
% 
%                       %% sample the weights from the new fitted weights in the other script (modelling KC_PnWeights.m)
% 
%                      
%                       thisW_HomogModel(whichPN_homog,k)= thisWeightHomo+ thisW_HomogModel(whichPN_homog,k); 
% 
% 
%                     end
%                 end 
%                 

% for trial = 1:(numm)
% 
%      Activations(:,:,trial) = thisW'*PNs(:,:,trial+numtrainingSamples);
%      Y(:,:,trial)=(( Activations(:,:,trial)-(InhibitionGain)/(n)*repmat(sum(Activations(:,:,trial),1),n,1)-thetaS')>0 ).*( Activations(:,:,trial)-InhibitionGain/(n)*repmat(sum(Activations(:,:,trial),1),n,1)-thetaS');
%     
%      ActivationsEqualized(:,:,trial) = thisW_equalizedModel'*PNs(:,:,trial+numtrainingSamples);
%      YEqualized(:,:,trial)=(( ActivationsEqualized(:,:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsEqualized(:,:,trial),1),n,1)-theta')>0 ).*( ActivationsEqualized(:,:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsEqualized(:,:,trial),1),n,1)-theta');
%      
%      
%     ActivationsHomog(:,:,trial) = thisW_HomogModel'*PNs(:,:,trial+numtrainingSamples);
%     YHomog(:,:,trial)=(( ActivationsHomog(:,:,trial)-(InhibitionGain)/(n)*repmat(sum(ActivationsHomog(:,:,trial),1),n,1)-thetaH')>0 ).*( ActivationsHomog(:,:,trial)-InhibitionGain/(n)*repmat(sum(ActivationsHomog(:,:,trial),1),n,1)-thetaH');
%                    
%      
% end


p_ra=0;
p_raEq=0;
p_raH=0;

odors= (size(Y,2))*numm;

for trials=numtrainingSamples+1:numTrials

                    for odour=1:(size(Y,2))

                        z1=Wop(:,1)'*Y(:,odour,trials);
                        z2=Wop(:,2)'*Y(:,odour,trials);

                        z1Eq=WopEq(:,1)'*YEqualized(:,odour,trials);
                        z2Eq=WopEq(:,2)'*YEqualized(:,odour,trials);

                        z1H=WopHom(:,1)'*YHomog(:,odour,trials);
                        z2H=WopHom(:,2)'*YHomog(:,odour,trials);

                        if (~ isempty(find(classAction1==odour)) )
                            
                            
                            pr_action1=  exp(C*z1)/(exp(C*z1)+exp(C*z2));
                            
                            %if(pr_action1>rand(1))
                            %/odors
                            p_ra=p_ra+(pr_action1/odors);
                            %end
                        end

                        if ( isempty((find(classAction1==odour))) )
                            
                            
                            pr_action2=  exp(C*z2)/(exp(C*z1)+exp(C*z2));
                            
                            
                           % if(pr_action2>rand(1))
                            p_ra=p_ra+(pr_action2/odors);
                            %end
                        
                        end


                        %% prob. of success in equalized model
                        if ( ~ isempty(find(classAction1==odour)) )
                           
                            pr_action1Eq=  exp(C*z1Eq)/(exp(C*z1Eq)+exp(C*z2Eq));
                            
                            %if(pr_action1Eq>rand(1))
                                p_raEq=p_raEq+(pr_action1Eq/odors);
                            %end
                        end

                        if ( isempty((find(classAction1==odour))) )
                            
                            pr_action2Eq=  exp(C*z2Eq)/(exp(C*z1Eq)+exp(C*z2Eq));
                            
                            %if(pr_action2Eq>rand(1))
                                p_raEq=p_raEq+(pr_action2Eq/odors);
                            %end
                        end
                        
                        %% prob. of success in the homogenous model 
                        if ( ~ isempty(find(classAction1==odour)) )
                            
                            pr_action1H=  exp(C*z1H)/(exp(C*z1H)+exp(C*z2H));
                            
                            %if(pr_action1H>rand(1))
                            p_raH=p_raH+(pr_action1H/odors);
                            %end
                        end

                        if (  isempty((find(classAction1==odour))) )
                            
                            pr_action2H=  exp(C*z2H)/(exp(C*z1H)+exp(C*z2H));
                            %if(pr_action2H>rand(1))
                                p_raH=p_raH+(pr_action2H/odors);
                            %end
                        end
                        
                        

                       
                        
                        
                        
                    end
end




end