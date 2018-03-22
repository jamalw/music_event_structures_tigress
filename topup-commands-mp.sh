#!/bin/bash

#name job pythondemo, output to slurm file, use partition all, run for 1500 minutes and use 30GB of ram
#SBATCH -J 'applywarp'
#SBATCH -o logfiles/applywarp-%j.out
#SBATCH --error=logfiles/applywarp_%j.err
#SBATCH -p all
#SBATCH -t 1500
#SBATCH -c 4 --mem 40000
#SBATCH --mail-type ALL
#SBATCH --mail-user jamalw@princeton.edu

module load fsl/5.0.9
module load afni/openmp.latest

# Merge the 3 AP volumes and the 3 PA volumes
fslmerge \
-t \
all_SE \
field_AP.nii \
field_PA.nii

# Run topup on the all_SE file
topup \
--imain=all_SE.nii.gz \
--datain=acqparams.txt \
--config=b02b0.cnf \
--out=topup_output \
--iout=topup_iout \
--fout=topup_fout \
--logout=topup_logout

# Create a magnitude image by averaging all 6 corrected SE volumes 
# (the topup_iout file).
fslsplit topup_iout iout-split -t
fslmaths iout-split0000 -add iout-split0001 -add iout-split0002 -add iout-split0003 -add iout-split0004 -add iout-split0005 -div 6 magnitude

# Create a magnitude_brain image as well with bet.
fslmaths topup_iout -Tmean magnitude
bet magnitude magnitude_brain

# Convert the fieldmap (topup_fout) from Hz to rad/s units:
fslmaths topup_fout -mul 6.28 topup_fout_rads

#First convert the topup-derived phase map (topup_fout) into a pixel-shift map.
fslmaths topup_fout.nii.gz -mul 0.0597 shiftmap.nii.gz

#Next convert the pixel-shift map into a deformation (warp) field (use the magnitude image as a reference).
convertwarp -r magnitude.nii.gz -s shiftmap.nii.gz -o warpfield

#Before we apply the warp to the EPI data, we will perform a motion correction to a reference EPI acquired close in time to the spin echoes. Extract the reference EPI volume (should be the same for all your EPI runs):
fslroi epi2.nii refvol.nii.gz 2520 1

#Run mcflirt on your EPI run:
mcflirt -in epi1.nii -out EPI_mcf1 -mats -reffile refvol.nii.gz -spline_final
mcflirt -in epi2.nii -out EPI_mcf2 -mats -reffile refvol.nii.gz -spline_final

#Mcflirt will output a transform matrix file for each EPI volume inside an outputfilename.mat folder. So next we will concatenate those into a single file:
cat EPI_mcf1.mat/MAT* > EPI_mcf1.cat
cat EPI_mcf2.mat/MAT* > EPI_mcf2.cat

#You can delete the motion-corrected EPI data that was output by mcflirt:
rm EPI_mcf1.nii.gz
rm EPI_mcf2.nii.gz

#Finally, apply the B0 correction along with the motion correction to the original EPI run:
applywarp -i epi1.nii -o EPI-corr-mc1 -r magnitude.nii.gz -w warpfield.nii.gz --premat=EPI_mcf1.cat --interp=spline --paddingsize=1

applywarp -i epi2.nii -o EPI-corr-mc2 -r magnitude.nii.gz -w warpfield.nii.gz --premat=EPI_mcf2.cat --interp=spline --paddingsize=1

