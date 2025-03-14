#!/usr/bin/env python3
# Methylation_Interplay_in_Genomic_Duplications.py
# A tool to analyze the bidirectional influence between 
# methylation patterns in duplicate regions and their surrounding genomic landscape

import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze DNA methylation of duplicate pairs and their adjacent regions.')
    parser.add_argument('--duplicates', required=True, help='File containing duplicate pairs coordinates')
    parser.add_argument('--methylation', required=True, help='File containing CpG methylation data')
    parser.add_argument('--output', required=True, help='Output file name')
    parser.add_argument('--min-distance', type=int, default=0, 
                        help='Minimum distance (bp) between duplicate pairs to be considered (default: 0)')
    parser.add_argument('--isolated', action='store_true', 
                        help='Filter out duplicate pairs that have other nearby duplicate regions')
    parser.add_argument('--isolation-distance', type=int, default=10000,
                        help='Minimum distance (bp) to other duplicate regions to be considered isolated (default: 10000)')
    parser.add_argument('--min-length', type=int, default=0,
                        help='Minimum length (bp) of duplicate regions to be included (default: 0)')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum length (bp) of duplicate regions to be included (default: no limit)')
    parser.add_argument('--length-difference', type=float, default=None,
                        help='Maximum allowed length difference between pairs as a fraction (e.g., 0.1 for 10%) (default: no limit)')
    return parser.parse_args()

def load_duplicate_pairs(duplicate_file):
    """
    Load duplicate pairs from file.
    Expected format: chr1, start1, end1, strand1, chr2, start2, end2, strand2, pair_id
    """
    columns = ['chr1', 'start1', 'end1', 'strand1', 'chr2', 'start2', 'end2', 'strand2', 'pair_id']
    try:
        df = pd.read_csv(duplicate_file, sep='\t', header=None, names=columns)
        # Convert positions to integers
        for col in ['start1', 'end1', 'start2', 'end2']:
            df[col] = df[col].astype(int)
        return df
    except Exception as e:
        print(f"Error loading duplicate pairs file: {e}")
        exit(1)

def load_methylation_data(methylation_file):
    """
    Load methylation data from file.
    Expected format: chr, start, end, methylated_reads, total_reads
    """
    columns = ['chr', 'start', 'end', 'methylated_reads', 'total_reads']
    try:
        df = pd.read_csv(methylation_file, sep='\t', header=None, names=columns)
        # Convert positions to integers
        for col in ['start', 'end', 'methylated_reads', 'total_reads']:
            df[col] = df[col].astype(int)
        # Calculate methylation rate for each CpG site
        df['methylation_rate'] = df['methylated_reads'] / df['total_reads']
        return df
    except Exception as e:
        print(f"Error loading methylation data file: {e}")
        exit(1)

def get_region_methylation(methyl_df, chrom, start, end):
    """
    Calculate average methylation for a genomic region.
    Returns tuple of (average_methylation, number_of_cpg_sites)
    """
    region_methyl = methyl_df[(methyl_df['chr'] == chrom) & 
                              (methyl_df['start'] >= start) & 
                              (methyl_df['end'] <= end)]
    
    if len(region_methyl) == 0:
        return 0, 0
    
    # Calculate weighted average methylation based on read counts
    total_methylated = region_methyl['methylated_reads'].sum()
    total_reads = region_methyl['total_reads'].sum()
    
    if total_reads == 0:
        return 0, len(region_methyl)
    
    avg_methylation = total_methylated / total_reads
    return avg_methylation, len(region_methyl)

def analyze_duplicate_pairs(duplicate_df, methyl_df):
    """
    Analyze methylation levels for duplicate pairs and their adjacent regions.
    Examines adjacent regions at multiple distances (1kb, 5kb, 10kb, 50kb, 100kb, 500kb).
    """
    results = []
    
    # Define the distance intervals to check (in kb)
    distance_intervals = [1, 5, 10, 50, 100, 500]
    
    for _, row in duplicate_df.iterrows():
        pair_id = row['pair_id']
        
        # Calculate lengths for duplicate regions
        length1 = row['end1'] - row['start1']
        length2 = row['end2'] - row['start2']
        
        # Get methylation for the duplicate regions themselves
        pair1_methyl, pair1_cpgs = get_region_methylation(methyl_df, row['chr1'], row['start1'], row['end1'])
        pair2_methyl, pair2_cpgs = get_region_methylation(methyl_df, row['chr2'], row['start2'], row['end2'])
        
        # Calculate methylation difference between pairs
        methyl_diff = abs(pair1_methyl - pair2_methyl)
        
        # Initialize result dictionary with basic information
        result = {
            'pair_id': pair_id,
            'chr1': row['chr1'],
            'start1': row['start1'],
            'end1': row['end1'],
            'length1': length1,
            'strand1': row['strand1'],
            'methyl1': pair1_methyl,
            'cpg_count1': pair1_cpgs,
            
            'chr2': row['chr2'],
            'start2': row['start2'],
            'end2': row['end2'],
            'length2': length2,
            'strand2': row['strand2'],
            'methyl2': pair2_methyl,
            'cpg_count2': pair2_cpgs,
            
            'methyl_diff': methyl_diff,
        }
        
        # Analyze adjacent regions at multiple distances
        for dist_kb in distance_intervals:
            dist_bp = dist_kb * 1000
            
            # Define regions for pair 1 based on strand
            if row['strand1'] == '+':
                # For positive strand, upstream is before start, downstream is after end
                upstream1_start = max(0, row['start1'] - dist_bp)
                upstream1_end = row['start1']
                downstream1_start = row['end1']
                downstream1_end = row['end1'] + dist_bp
            else:
                # For negative strand, upstream is after end, downstream is before start
                upstream1_start = row['end1']
                upstream1_end = row['end1'] + dist_bp
                downstream1_start = max(0, row['start1'] - dist_bp)
                downstream1_end = row['start1']
            
            # Define regions for pair 2 based on strand
            if row['strand2'] == '+':
                # For positive strand, upstream is before start, downstream is after end
                upstream2_start = max(0, row['start2'] - dist_bp)
                upstream2_end = row['start2']
                downstream2_start = row['end2']
                downstream2_end = row['end2'] + dist_bp
            else:
                # For negative strand, upstream is after end, downstream is before start
                upstream2_start = row['end2']
                upstream2_end = row['end2'] + dist_bp
                downstream2_start = max(0, row['start2'] - dist_bp)
                downstream2_end = row['start2']
            
            # Calculate methylation for each region
            upstream1_methyl, up1_cpgs = get_region_methylation(methyl_df, row['chr1'], upstream1_start, upstream1_end)
            downstream1_methyl, down1_cpgs = get_region_methylation(methyl_df, row['chr1'], downstream1_start, downstream1_end)
            
            upstream2_methyl, up2_cpgs = get_region_methylation(methyl_df, row['chr2'], upstream2_start, upstream2_end)
            downstream2_methyl, down2_cpgs = get_region_methylation(methyl_df, row['chr2'], downstream2_start, downstream2_end)
            
            # Calculate methylation differences between corresponding regions
            upstream_methyl_diff = abs(upstream1_methyl - upstream2_methyl)
            downstream_methyl_diff = abs(downstream1_methyl - downstream2_methyl)
            
            # Add results for this distance to the result dictionary
            result.update({
                f'upstream1_methyl_{dist_kb}kb': upstream1_methyl,
                f'upstream1_cpgs_{dist_kb}kb': up1_cpgs,
                f'downstream1_methyl_{dist_kb}kb': downstream1_methyl,
                f'downstream1_cpgs_{dist_kb}kb': down1_cpgs,
                
                f'upstream2_methyl_{dist_kb}kb': upstream2_methyl,
                f'upstream2_cpgs_{dist_kb}kb': up2_cpgs,
                f'downstream2_methyl_{dist_kb}kb': downstream2_methyl,
                f'downstream2_cpgs_{dist_kb}kb': down2_cpgs,
                
                f'upstream_methyl_diff_{dist_kb}kb': upstream_methyl_diff,
                f'downstream_methyl_diff_{dist_kb}kb': downstream_methyl_diff,
            })
        
        results.append(result)
    
    return pd.DataFrame(results)

def filter_duplicate_pairs(duplicate_df, min_distance=0, isolated=False, isolation_distance=10000,
                        min_length=0, max_length=None, length_difference=None):
    """
    Filter duplicate pairs based on distance and length criteria.
    
    Parameters:
    -----------
    duplicate_df : pandas.DataFrame
        DataFrame containing duplicate pairs.
    min_distance : int
        Minimum distance between duplicate pairs to be considered.
    isolated : bool
        Whether to filter for isolated duplicate pairs.
    isolation_distance : int
        Minimum distance to other duplicate regions to be considered isolated.
    min_length : int
        Minimum length of duplicate regions to be included.
    max_length : int or None
        Maximum length of duplicate regions to be included.
    length_difference : float or None
        Maximum allowed length difference between pairs as a fraction.
        
    Returns:
    --------
    pandas.DataFrame
        Filtered duplicate pairs.
    """
    original_count = len(duplicate_df)
    filtered_stats = {}
    
    # Calculate region lengths
    duplicate_df['length1'] = duplicate_df['end1'] - duplicate_df['start1']
    duplicate_df['length2'] = duplicate_df['end2'] - duplicate_df['start2']
    
    # Filter by minimum length
    if min_length > 0:
        before_count = len(duplicate_df)
        duplicate_df = duplicate_df[(duplicate_df['length1'] >= min_length) & (duplicate_df['length2'] >= min_length)]
        filtered_stats['min_length'] = before_count - len(duplicate_df)
    
    # Filter by maximum length
    if max_length is not None:
        before_count = len(duplicate_df)
        duplicate_df = duplicate_df[(duplicate_df['length1'] <= max_length) & (duplicate_df['length2'] <= max_length)]
        filtered_stats['max_length'] = before_count - len(duplicate_df)
    
    # Filter by length difference
    if length_difference is not None:
        before_count = len(duplicate_df)
        # Calculate fractional length difference
        duplicate_df['length_diff_ratio'] = duplicate_df.apply(
            lambda row: abs(row['length1'] - row['length2']) / max(row['length1'], row['length2']),
            axis=1
        )
        duplicate_df = duplicate_df[duplicate_df['length_diff_ratio'] <= length_difference]
        filtered_stats['length_diff'] = before_count - len(duplicate_df)
    
    # Filter out duplicate pairs that are too close to each other
    if min_distance > 0:
        before_count = len(duplicate_df)
        # Calculate distance between pair1 and pair2
        duplicate_df['distance'] = 0
        
        # For pairs on the same chromosome, calculate the minimum distance between any endpoints
        same_chrom_mask = duplicate_df['chr1'] == duplicate_df['chr2']
        
        for idx, row in duplicate_df[same_chrom_mask].iterrows():
            # Calculate all possible distances between the regions
            d1 = abs(row['end1'] - row['start2'])  # end1 to start2
            d2 = abs(row['start1'] - row['end2'])  # start1 to end2
            d3 = abs(row['start1'] - row['start2'])  # start1 to start2
            d4 = abs(row['end1'] - row['end2'])  # end1 to end2
            duplicate_df.at[idx, 'distance'] = min(d1, d2, d3, d4)
        
        # Filter based on minimum distance
        duplicate_df = duplicate_df[(~same_chrom_mask) | (duplicate_df['distance'] >= min_distance)]
        filtered_stats['min_distance'] = before_count - len(duplicate_df)
    
    # Filter for isolated duplicate pairs
    if isolated:
        before_count = len(duplicate_df)
        # Create a list of all duplicate regions
        all_regions = []
        for _, row in duplicate_df.iterrows():
            all_regions.append((row['chr1'], row['start1'], row['end1'], row['pair_id']))
            all_regions.append((row['chr2'], row['start2'], row['end2'], row['pair_id']))
        
        regions_df = pd.DataFrame(all_regions, columns=['chr', 'start', 'end', 'pair_id'])
        
        # Filter out pairs that have other duplicate regions nearby
        isolated_pairs = set()
        
        for _, row in duplicate_df.iterrows():
            pair_id = row['pair_id']
            is_isolated = True
            
            # Check if pair1 has any nearby duplicate regions (excluding its own pair)
            nearby_to_pair1 = regions_df[
                (regions_df['chr'] == row['chr1']) & 
                (regions_df['pair_id'] != pair_id) &
                (
                    ((regions_df['start'] >= row['start1'] - isolation_distance) & 
                     (regions_df['start'] <= row['end1'] + isolation_distance)) |
                    ((regions_df['end'] >= row['start1'] - isolation_distance) & 
                     (regions_df['end'] <= row['end1'] + isolation_distance))
                )
            ]
            
            # Check if pair2 has any nearby duplicate regions (excluding its own pair)
            nearby_to_pair2 = regions_df[
                (regions_df['chr'] == row['chr2']) & 
                (regions_df['pair_id'] != pair_id) &
                (
                    ((regions_df['start'] >= row['start2'] - isolation_distance) & 
                     (regions_df['start'] <= row['end2'] + isolation_distance)) |
                    ((regions_df['end'] >= row['start2'] - isolation_distance) & 
                     (regions_df['end'] <= row['end2'] + isolation_distance))
                )
            ]
            
            if len(nearby_to_pair1) == 0 and len(nearby_to_pair2) == 0:
                isolated_pairs.add(pair_id)
        
        # Filter the dataframe to only include isolated pairs
        duplicate_df = duplicate_df[duplicate_df['pair_id'].isin(isolated_pairs)]
        filtered_stats['isolated'] = before_count - len(duplicate_df)
    
    filtered_count = len(duplicate_df)
    total_filtered = original_count - filtered_count
    print(f"Filtered {total_filtered} duplicate pairs based on filtering criteria")
    
    # Print detailed filtering statistics
    if filtered_stats:
        print("Filtering statistics:")
        for filter_name, count in filtered_stats.items():
            print(f"  - {filter_name}: {count} pairs filtered")
    
    return duplicate_df

def main():
    args = parse_arguments()
    
    print(f"Loading duplicate pairs from {args.duplicates}")
    duplicate_df = load_duplicate_pairs(args.duplicates)
    print(f"Loaded {len(duplicate_df)} duplicate pairs")
    
    # Apply filters if specified
    duplicate_df = filter_duplicate_pairs(
        duplicate_df, 
        min_distance=args.min_distance,
        isolated=args.isolated,
        isolation_distance=args.isolation_distance,
        min_length=args.min_length,
        max_length=args.max_length,
        length_difference=args.length_difference
    )
    print(f"Retained {len(duplicate_df)} duplicate pairs after filtering")
    
    print(f"Loading methylation data from {args.methylation}")
    methyl_df = load_methylation_data(args.methylation)
    print(f"Loaded {len(methyl_df)} CpG sites")
    
    print("Analyzing methylation in duplicate pairs and adjacent regions...")
    results_df = analyze_duplicate_pairs(duplicate_df, methyl_df)
    
    print(f"Writing results to {args.output}")
    results_df.to_csv(args.output, sep='\t', index=False)
    
    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Total duplicate pairs analyzed: {len(results_df)}")
    
    avg_methyl_diff = results_df['methyl_diff'].mean()
    print(f"Average methylation difference between duplicate pairs: {avg_methyl_diff:.4f}")
    
    # Distance-based analysis - show correlation between distances and methylation differences
    distance_intervals = [1, 5, 10, 50, 100, 500]
    print("\nMethylation differences across distance intervals:")
    
    print("\nUpstream regions:")
    for dist_kb in distance_intervals:
        avg_diff = results_df[f'upstream_methyl_diff_{dist_kb}kb'].mean()
        print(f"  {dist_kb} kb: {avg_diff:.4f}")
    
    print("\nDownstream regions:")
    for dist_kb in distance_intervals:
        avg_diff = results_df[f'downstream_methyl_diff_{dist_kb}kb'].mean()
        print(f"  {dist_kb} kb: {avg_diff:.4f}")
    
    # Calculate correlations between methylation differences in duplicate pairs and their adjacent regions
    print("\nCorrelations between methylation differences:")
    for dist_kb in distance_intervals:
        # Calculate correlation coefficients
        pair_vs_upstream = np.corrcoef(
            results_df['methyl_diff'].dropna(),
            results_df[f'upstream_methyl_diff_{dist_kb}kb'].dropna()
        )[0, 1]
        
        pair_vs_downstream = np.corrcoef(
            results_df['methyl_diff'].dropna(),
            results_df[f'downstream_methyl_diff_{dist_kb}kb'].dropna()
        )[0, 1]
        
        print(f"  {dist_kb} kb upstream correlation: {pair_vs_upstream:.4f}")
        print(f"  {dist_kb} kb downstream correlation: {pair_vs_downstream:.4f}")

if __name__ == "__main__":
    main()
