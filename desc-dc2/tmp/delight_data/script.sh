#!/bin/bash

if  [ -f "galaxies-fluxredshifts2-Sens.txt" ] ; then
  rm galaxies-fluxredshifts2-Sens.txt
fi

touch galaxies-fluxredshifts2-Sens.txt
for file in `ls galaxies-fluxredshifts2_*.txt` ; do
  head $file >> galaxies-fluxredshifts2-Sens.txt
  tail $file >> galaxies-fluxredshifts2-Sens.txt
done


if  [ -f "galaxies-redshiftpdfs-Sens.txt" ] ; then
  rm galaxies-redshiftpdfs-Sens.txt
fi

touch galaxies-redshiftpdfs-Sens.txt
for file in `ls galaxies-redshiftpdfs_*.txt` ; do
  head $file >> galaxies-redshiftpdfs-Sens.txt
  tail $file >> galaxies-redshiftpdfs-Sens.txt
done


if  [ -f "galaxies-redshiftpdfs-cww-Sens.txt" ] ; then
  rm galaxies-redshiftpdfs-cww-Sens.txt
fi

touch galaxies-redshiftpdfs-cww-Sens.txt
for file in `ls galaxies-redshiftpdfs-cww_*.txt` ; do
  head $file >> galaxies-redshiftpdfs-cww-Sens.txt
  tail $file >> galaxies-redshiftpdfs-cww-Sens.txt
done


if  [ -f "galaxies-redshiftmetrics-Sens.txt" ] ; then
  rm galaxies-redshiftmetrics-Sens.txt
fi

touch galaxies-redshiftmetrics-Sens.txt
for file in `ls galaxies-redshiftmetrics_*.txt` ; do
  head $file >> galaxies-redshiftmetrics-Sens.txt
  tail $file >> galaxies-redshiftmetrics-Sens.txt
done


if  [ -f "galaxies-redshiftmetrics-cww-Sens.txt" ] ; then
  rm galaxies-redshiftmetrics-cww-Sens.txt
fi

touch galaxies-redshiftmetrics-cww-Sens.txt
for file in `ls galaxies-redshiftmetrics-cww_*.txt` ; do
  head $file >> galaxies-redshiftmetrics-cww-Sens.txt
  tail $file >> galaxies-redshiftmetrics-cww-Sens.txt
done

