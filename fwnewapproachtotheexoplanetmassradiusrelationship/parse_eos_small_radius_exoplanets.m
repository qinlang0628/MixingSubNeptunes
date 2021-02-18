%parse_eos_small_radius_exoplanets

%Purpose: Parse ANEOS for H2O, and Saumon-Chabrier EOS for H2

%Parse ANEOS for H2O; - email of Nov 7, 2016, 6:55 AM from <federico.benitez@space.unibe.ch>
%log10(T),log10(P),RHO,U,S,CV,CP,DPDT,DPDRHO,DeltaADIAB,FKRO,CS,KPA,R2PL,R2PH,ZBAR,Cp/Cv,adexpID,mmw

% d = fopen('H2O/H2O_MANEOS.dat');
% tline = fgetl(d);
% i = 0;
% while ischar(tline)
%   i = i + 1;
%   if rem(i,1e4) == 1; disp(i);end
%   if contains(tline,'Infinity') == 0
%     h(i,:) = sscanf(tline,'%f');
%   end
%   tline = fgetl(d);
% end
% save H2O_MANEOS_DATA
% fclose(d)

%Saumon-Chabrier for H2
for neos = 1:2
    
switch neos
    case 1
        %All logarithms are in base 10 and the contribution from the photon gas is NOT included
        d = fopen('Saumon95_I/H_TAB_I.A'); %see README_I for details
    case 2
        d = fopen('Saumon95_I/HE_TAB_I.A');
end

tline = fgetl(d);
tindex = 0;

while ischar(tline)
  tline = fgetl(d);
  if length(tline)<30 & tline>-1  %A new block of good data has started. Reset the counter
     tindex = tindex + 1;
     T(tindex) = str2num(tline(2:5));
     ntoread(tindex) = str2num(tline(8:9)); %number of pressure points for this T
     for n = 1:ntoread(tindex)
         tline=fgetl(d);
         h(tindex,n,1:11) = sscanf(tline,'%f');
         h(tindex,n,12) = 10.^T(tindex);
     end
  end
end

%Unit conversions:
T = 10.^T;
h(h==0) = NaN;
h(:,:,1) = (10.^h(:,:,1))/10; %dyn/cm^2 --> Pa
h(:,:,4) = (10.^(h(:,:,4)))*1e3; %g/cm^3 --> kg/m^3
%h(:,:,5), i.e.  entropy, is in erg/g/K
%h(:,:,10) is the adiabatic gradianet (d log T / d log P)

fclose(d)

switch neos
    case 1
        HEOS = h;
    case 2
        HEEOS = h;
end

end

%H/He mixture, following Section 6 of Saumon et al. 1995:
solarEOS = 0.25/4;

%Compute fugacity coefficients following email to Fegley on Oct 2, 2019
V_ideal = 1*8.314*repmat(T,76,1)'./HEOS(:,:,1); %per mole
rho_ideal = 1./V_ideal*0.002; %molecular hydrogen
ZH = rho_ideal./HEOS(:,:,4); %Z = compression factor (high for less compressible, lower density)

%Integrate from 100 bars to minimize effect of dissociation and ionization
for Ti = 1:62
 %Densify before cumulative summation
 lastgood = max(find(isnan(HEOS(Ti,:,1))==0));
 dense_p(Ti,:) = logspace(log10(1e3),log10(max(HEOS(Ti,21:lastgood,1))),5000); %minimize discretization error
 dense_V_ideal(Ti,:) = 1*8.314*squeeze(repmat(T(Ti),5000,1))./dense_p(Ti,:)';
 dense_rho_ideal(Ti,:) = 1./dense_V_ideal(Ti,:)*0.002; %molecular hydrogen
 dense_rho(Ti,:) = interp1(HEOS(Ti,1:lastgood,1),HEOS(Ti,1:lastgood,4),dense_p(Ti,:));
 dense_Z(Ti,:) = dense_rho_ideal(Ti,:)./dense_rho(Ti,:);
 dense_Z(Ti,:) = (dense_Z(Ti,:)-1) + 1;
 %dense_Z2(Ti,:) =
 %interp1(HEOS(Ti,1:lastgood,1),ZH(Ti,1:lastgood),dense_p(Ti,:));
 %%deprecated because leads to less-smooth plots
 dense_Z(Ti,1:3000) = 1; %(Ti,1:2704) = 1;
 dense_ln_phi_H(Ti,:) = cumsum(( ((dense_Z(Ti,1:(end-1)))-1)./dense_p(Ti,1:(end-1)) ).* diff(dense_p(Ti,:)));
 %ln_phi_H is deprecated because the gap between the 
 %ln_phi_H(Ti,1:75)   = cumsum((ZH(Ti,1:(end-1))-1)./HEOS(Ti,1:(end-1),1).*diff(HEOS(Ti,1:end,1)));
end


save Saumon95_fugacity  T HEOS dense_p dense_ln_phi_H ZH %ln_phi_H 

V_ideal = 1*8.314*repmat(T,76,1)'./HEEOS(:,:,1); %per mole
rho_ideal = 1./V_ideal*0.004; %atomic helium
ZHE = rho_ideal./HEEOS(:,:,4); %Z = compression factor (high for less compressible, lower density)


%Drop adiabats from an RCB (radiative-convective boundary). We want density
%and temperature as a function of pressure along the adiabat.
%Constant-gravity assumption.

figure
P_RCB = 1e5; %Pa. This is an underestimate of the true P_RCB for sub-Neptunes.
adindex = 1;
for Tinit = [160,250:250:1500]; %K (close-in sub-Neptune)
%Figure out what the entropy (S) is for this starting P and T by linear
%interpolation in log P and log T.
Tpane_low = max(find(T<Tinit))
Tpane_high = Tpane_low + 1;

%In the EOS, pressures are the same for each T-pane for low T, although
%higher T panes have additional P rows
p_low = max(find(HEOS(Tpane_low,:,1)<P_RCB))
p_high = p_low + 1;

my_S_Tpane_low = interp1(log10(HEOS(Tpane_low,1:ntoread(Tpane_low),1)), ...
                         HEOS(Tpane_low,1:ntoread(Tpane_low),5),log10(P_RCB)); %isentrope
my_S_Tpane_high = interp1(log10(HEOS(Tpane_high,1:ntoread(Tpane_low),1)), ...
                         HEOS(Tpane_high,1:ntoread(Tpane_low),5),log10(P_RCB)); %isentrope
my_S = interp1(log10([T(Tpane_low),T(Tpane_high)]),[my_S_Tpane_low, my_S_Tpane_high],log10(Tinit))

%Now for each T-pane, interpolate the entropy we found above.
%Note that entropy is already stored as log so linear interpolation is OK
for l = 1:size(HEOS,1)
    p_low = max(find(HEOS(l,1:ntoread(l),5)>my_S)); %increase p, decrease S
    if p_low < size(HEOS,2)
    if isnan(HEOS(l,p_low+1,5)) == 0 %very low T-panes may lack the entropies relevant for close-in exoplanet atmospheres
      my_mix = interp1([HEOS(l,p_low,5),HEOS(l,p_low+1,5)],[0,1],my_S);
      my_adiabat(adindex,l,:) = HEOS(l,p_low,:)*(1-my_mix) + HEOS(l,p_low,:)*my_mix;
    end
    end
end

plot(squeeze(my_adiabat(adindex,:,1)),T(1:size(my_adiabat,2)),'r','LineWidth',adindex)
set(gca,'xscale','log')
set(gca,'yscale','log')
hold on
plot(squeeze(my_adiabat(adindex,:,1)),squeeze(my_adiabat(adindex,:,4)),'k','LineWidth',adindex) %density

%Overplot ideal gas law
V_ideal = 1*8.314*T(1:size(my_adiabat,2))./squeeze(my_adiabat(adindex,:,1));
rho_ideal = 1./V_ideal*0.002; %molecular hydrogen
plot(squeeze(my_adiabat(adindex,:,1)),rho_ideal,'Color','k','LineWidth',adindex,'LineStyle','--')

adindex = adindex + 1;

line([1e6 1e10], [Tinit Tinit*1e4^0.2222]);
line([1e6 1e10], [Tinit Tinit*1e4^0.25]);


end

ylabel('red is temperature, black is density')


grid on

%Figure the radius boost along each adiabat for 1 REarth, neglecting
%gravity of the envelope, as a function of base-of-envelope pressure.
G = 6.67e-11; %mks
CoreMassEarths = 5;
R_pl = 6378000*CoreMassEarths^0.27; %1 Earth radius
M_pl = 6e24*CoreMassEarths; 

boep_list = [2e8]; %base-of-envelope-pressure (Pa);
dr = 1e4; %m
for adindex = 1:size(my_adiabat,1) %RCB adiabatic temperature index
    for l = 1:length(boep_list)
        r = 0; i = 1;
        pr(adindex,l,i) = boep_list(l);
        eoa = max(find(my_adiabat(adindex,:,1)>0)); %end of adiabat
        boa = min(find(my_adiabat(adindex,:,1)>0)); %beginning of adiabat
        rhob(adindex,l,i) = interp1(squeeze(my_adiabat(adindex,boa:eoa,1)),squeeze(my_adiabat(adindex,boa:eoa,4)),pr(adindex,l,i));
        while pr(adindex,l,i) > 1e5 %pressure as a function of radius
            i = i + 1;
            g(adindex,l,i) = G*M_pl/((R_pl + r).^2); 
            pr(adindex,l,i) = pr(adindex,l,i-1) - dr*rhob(adindex,l,i-1)*g(adindex,l,i);
            r = r + dr;
            rhob(adindex,l,i) = ...
             interp1(squeeze(my_adiabat(adindex,boa:eoa,1)),squeeze(my_adiabat(adindex,boa:eoa,4)),pr(adindex,l,i));
            Tb(adindex,l,i) = ...
             interp1(squeeze(my_adiabat(adindex,boa:eoa,1)),T(boa:eoa),pr(adindex,l,i));
            mshellb(adindex,l,i) = ...
             4*pi*((R_pl + r).^2)*dr*rhob(adindex,l,i-1); %kg
        end
    end
end
    
figure;plot(dr*[1:size(mshellb,3)],cumsum(squeeze(mshellb)'));
xlabel('Height above volatile-silicate interface (m)')
ylabel('Cumulative volatile mass below this level (kg)')
grid on

figure;plot(dr*[1:size(mshellb,3)],(squeeze(Tb)'));
xlabel('Height above volatile-silicate interface (m)')
ylabel('Temperature (K)')
grid on

