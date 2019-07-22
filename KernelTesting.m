function [p_ra,p_raEq,p_raH]=KernelTesting (C,Wop,WopEq,WopHom,PNs,PnToKc,HomogPnToKc,theta,thetaS,thetaH,InhibitionGain,classAction1,numTrials,numtrainingSamples,Y,YEqualized,YHomog)

load('PW_given_theta_and_n.mat');
load('W_PN_KC.mat');
load ('P_n.mat');

numm=numTrials-numtrainingSamples;

n=2000;


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

%                             if(isnan(p_ra+(pr_action1/odors)))
%                                 stop=1; 
%                             end
                           
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