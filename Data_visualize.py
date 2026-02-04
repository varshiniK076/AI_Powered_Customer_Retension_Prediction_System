import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('visualize')

class DATA_VISUALIZE_EDA():
    def plot(df):
        try:
            Plot = 'EDA_images'
            # --- PLOT - 1: Pie chart for Churn Distribution ---
            plt.figure(figsize=(6, 6))
            df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['teal', 'gold'])
            plt.title('Target Variable: Churn - Yes vs No')
            plt.ylabel('')
            plt.savefig(f'{Plot}/pie_chart.png')
            plt.close()

            # --- Plot 2: Gender -> Churn ---
            gender_churn = pd.crosstab(df['gender'], df['Churn'])

            plt.figure(figsize=(7,20))
            ax = gender_churn.plot(kind='bar')

            plt.title('Churn Distribution by Gender')
            plt.xlabel('Gender')
            plt.ylabel('Customer Count')
            plt.xticks(rotation=0)
            plt.legend(title='Churn')

            # Add values + Percentage on bars
            for container in ax.containers:
                total = sum(container.datavalues)

                for bar, value in zip(container, container.datavalues):
                    percentage = value / total * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f'{int(value)}\n({percentage:.1f}%)',
                        ha='center',
                        va='top'
                    )

            plt.tight_layout()
            plt.savefig(f'{Plot}/Gender_vs_Churn.png')
            plt.close()


            # --- Plot 3: Gender -> senior citizen -> churn ---
            grouped = pd.crosstab([df['gender'], df['SeniorCitizen']], df['Churn'])
            plt.figure(figsize=(9,9))
            ax = grouped.plot(kind='bar', width=0.7)
            ax = grouped.plot(kind='bar', width=0.7)
            plt.title('Churn by Gender and Senior Citizen')
            plt.xlabel('Gender and Senior Citizen')
            plt.ylabel('Customer Count')
            plt.xticks(rotation=0)
            plt.legend(title='Churn')
            ax.legend(title='Senior Citizen\n0 = No, 1 = Yes')
            # Add values + percentage on bars
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    # Total for this x-category (sum of stacked/grouped bars at this x)
                    total = sum([c.get_height() for c in container])
                    pct = (height / total * 100) if total > 0 else 0

                    # Percentage inside the bar
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,  # middle of the bar
                        f'{pct:.1f}%',
                        ha='center',
                        va='top',
                        color='black',  # good contrast inside bar
                        fontsize = 8
                    )

                    # Count on top of the bar
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 2,  # slightly above the bar
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            plt.tight_layout()
            plt.savefig(f'{Plot}/Gender - SeniorCitizen - Churn.png')
            plt.close()

            # --- Plot 4: SIM Provider -> Senior Citizen by Gender ---

            fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharey=True)

            # ---- Male ----
            male_df = df[df['gender'] == 'Male']
            male_data = pd.crosstab(male_df['Service_Provider'], male_df['SeniorCitizen'])

            ax1 = male_data.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Male Customers')
            axes[0].set_xlabel('SIM Provider')
            axes[0].set_ylabel('Customer Count')
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
            axes[0].legend(title='Senior Citizen\n0 = No, 1 = Yes')

            for container in ax1.containers:
                ax1.bar_label(container, padding=1)

            # ---- Female ----
            female_df = df[df['gender'] == 'Female']
            female_data = pd.crosstab(female_df['Service_Provider'], female_df['SeniorCitizen'])

            ax2 = female_data.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Female Customers')
            axes[1].set_xlabel('SIM Provider')
            axes[1].set_ylabel('Customer Count')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
            axes[1].legend(title='Senior Citizen\n0 = No, 1 = Yes')

            for container in ax2.containers:
                ax2.bar_label(container, padding=1)

            # ---- Overall Title ----
            fig.suptitle('Senior Citizen Distribution by SIM Provider and Gender',fontsize=15)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'{Plot}/SIMProvider_Gender_SeniorCitizen_Subplot.png')
            plt.close()

            # --- Plot 5: Phone Service → Gender → Senior Citizen → Churn ---
            data = (
                df.groupby(['PhoneService', 'gender', 'SeniorCitizen', 'Churn'])
                .size()
                .unstack(fill_value=0)
            )

            fig, axes = plt.subplots(2, 1, figsize=(6,10), sharey=True)
            bar_width = 0.35

            for ax, phone in zip(axes, ['No', 'Yes']):
                temp = data.loc[phone]
                x = np.arange(len(temp))

                # Bars
                bars_no = ax.bar(x - bar_width / 2, temp['No'], bar_width, label='Churn = No')
                bars_yes = ax.bar(x + bar_width / 2, temp['Yes'], bar_width, label='Churn = Yes')

                # Add values on bars
                ax.bar_label(bars_no, padding=2, fontsize=8)
                ax.bar_label(bars_yes, padding=2, fontsize=8)

                # Axis formatting
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f'{g}\nSenior={s}' for g, s in temp.index],
                    rotation=0
                )
                ax.set_title(f'Phone Service = {phone}')
                ax.set_xlabel('Gender & Senior Citizen')

            axes[0].set_ylabel('Customer Count')
            axes[0].legend(title='Churn')
            axes[1].legend(title='Churn')

            plt.suptitle('Churn Analysis by Phone Service, Gender and Senior Citizen')

            plt.tight_layout()
            plt.savefig(f'{Plot}/PhoneService_Gender_SeniorCitizen_Churn.png')
            plt.close()

            # --- Plot 6: MultipleLines_Gender_SeniorCitizen_ServiceProvider ---
            providers = df['Service_Provider'].unique()

            fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
            axes = axes.flatten()

            bar_width = 0.25
            x = np.arange(4)  # Male-S0, Male-S1, Female-S0, Female-S1

            for ax, provider in zip(axes, providers):

                temp = df[df['Service_Provider'] == provider]
                data = temp.groupby(['gender', 'SeniorCitizen', 'MultipleLines']).size().unstack(fill_value=0)

                yes_vals, no_vals, no_phone_vals = [], [], []

                for gender in ['Male', 'Female']:
                    for sc in [0, 1]:
                        yes_vals.append(data.loc[(gender, sc), 'Yes'] if 'Yes' in data.columns else 0)
                        no_vals.append(data.loc[(gender, sc), 'No'] if 'No' in data.columns else 0)
                        no_phone_vals.append(
                            data.loc[(gender, sc), 'No phone service']
                            if 'No phone service' in data.columns else 0
                        )

                ax.bar(x - bar_width, yes_vals, width=bar_width, label='Multiple Lines = Yes')
                ax.bar(x, no_vals, width=bar_width, label='Multiple Lines = No')
                ax.bar(x + bar_width, no_phone_vals, width=bar_width, label='No Phone Service')

                # Add values on bars
                for bars in ax.containers:
                    ax.bar_label(bars, padding=2, fontsize=9)

                ax.set_title(f'Service Provider: {provider}')
                ax.set_xticks(x)
                ax.set_xticklabels([
                    'Male\nSenior=0', 'Male\nSenior=1',
                    'Female\nSenior=0', 'Female\nSenior=1'
                ])
                ax.set_xlabel('Gender & Senior Citizen')

            axes[0].set_ylabel('Customer Count')
            axes[0].legend(title='Multiple Lines')

            plt.suptitle(
                'Multiple Lines Distribution by Gender & Senior Citizen (Across Service Providers)',
                fontsize=15
            )
            plt.tight_layout()
            plt.savefig(f'{Plot}/MultipleLines_Grouped_ServiceProvider.png')
            plt.close()


            # Plot 7: Customer Counts & Percentage by Internet Type, Service Provider, Gender, and Senior Citizen ---

            # --- FOR DSL InternetService ---
            dsl_df = df[df['InternetService'] == 'DSL']

            # Crosstab: Service Provider → Gender vs Churn
            grouped = pd.crosstab(
                [dsl_df['Service_Provider'], dsl_df['gender']],
                dsl_df['Churn']
            )

            # Plot grouped bar chart
            ax = grouped.plot(kind='bar', figsize=(12, 7), width=0.8)

            plt.title('DSL → Service Provider → Gender → Churn')
            plt.xlabel('Service Provider / Gender')
            plt.ylabel('Customer Count')
            plt.xticks(rotation=0)
            plt.legend(title='Churn', loc='upper right')

            # Add percentage inside bars and count on top
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    # Total for this x-category (sum across churn 0/1)
                    idx = int(bar.get_x() + bar.get_width() / 2)
                    total = sum([c.get_height() for c in container])
                    pct = (height / total * 100) if total > 0 else 0

                    # Percentage inside bar
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,
                        f'{pct:.1f}%',
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=9
                    )

                    # Count on top
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 1,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            plt.tight_layout()
            plt.savefig(f'{Plot}/DSL_ServiceProvider_Gender_Churn.png')
            plt.close()

            # --- FOR Fiber Optic ---
            fiber_df = df[df['InternetService'] == 'Fiber optic']

            grouped = pd.crosstab(
                [fiber_df['Service_Provider'], fiber_df['gender']],
                fiber_df['Churn']
            )

            ax = grouped.plot(kind='bar', figsize=(12, 7), width=0.8, color=['teal', 'gold'])

            plt.title('Fiber optic → Service Provider → Gender → Churn')
            plt.xlabel('Service Provider / Gender')
            plt.ylabel('Customer Count')
            plt.xticks(rotation=0)
            plt.legend(title='Churn', loc='upper right')

            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    total = sum(c.get_height() for c in container)
                    pct = (height / total * 100) if total > 0 else 0

                    ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                            f'{pct:.1f}%', ha='center', va='center', fontsize=9)

                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(f'{Plot}/FiberOptic_ServiceProvider_Gender_Churn.png')
            plt.close()

            # --- For No Service ---
            no_df = df[df['InternetService'] == 'No']

            grouped = pd.crosstab(
                [no_df['Service_Provider'], no_df['gender']],
                no_df['Churn']
            )

            ax = grouped.plot(kind='bar', figsize=(16, 7), width=0.8, color=['teal', 'gold'])

            plt.title('No Internet → Service Provider → Gender → Churn')
            plt.xlabel('Service Provider / Gender')
            plt.ylabel('Customer Count')
            plt.xticks(rotation=0)
            plt.legend(title='Churn', loc='upper right')

            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    total = sum(c.get_height() for c in container)
                    pct = (height / total * 100) if total > 0 else 0

                    ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                            f'{pct:.1f}%', ha='center', va='center', fontsize=9)

                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(f'{Plot}/NoInternet_ServiceProvider_Gender_Churn.png')
            plt.close()

            # --- Plot 8: Contract vs Gender by Churn with SIM provider ---

            contracts = ['Month-to-month', 'One year', 'Two year']
            sims = df['Service_Provider'].unique()

            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
            axes = axes.flatten()  # flatten 2D array

            for ax, sim in zip(axes, sims):

                x = np.arange(len(contracts))

                # Prepare data
                total, mn, fn, my, fy = [], [], [], [], []

                for c in contracts:
                    temp = df[(df['Service_Provider'] == sim) & (df['Contract'] == c)]
                    total.append(len(temp))
                    mn.append(len(temp[(temp['gender'] == 'Male') & (temp['Churn'] == 'No')]))
                    fn.append(len(temp[(temp['gender'] == 'Female') & (temp['Churn'] == 'No')]))
                    my.append(len(temp[(temp['gender'] == 'Male') & (temp['Churn'] == 'Yes')]))
                    fy.append(len(temp[(temp['gender'] == 'Female') & (temp['Churn'] == 'Yes')]))

                # Background total bar
                ax.bar(x, total, width=0.6, color='#E6E6E6', edgecolor='gray')

                # Overlay bars
                b1 = ax.bar(x - 0.18, mn, width=0.15, color='teal', label='Male - No')
                b2 = ax.bar(x - 0.03, fn, width=0.15, color='gold', label='Female - No')
                b3 = ax.bar(x + 0.12, my, width=0.15, color='darkred', label='Male - Yes')
                b4 = ax.bar(x + 0.27, fy, width=0.15, color='orange', label='Female - Yes')

                # Add total values on top only
                for i, v in enumerate(total):
                    if v > 0:
                        ax.text(x[i], v + 3, v, ha='center', va='bottom', fontsize=10, fontweight='bold')

                ax.set_xticks(x)
                ax.set_xticklabels(contracts, rotation=0)
                ax.set_title(f'SIM = {sim}', fontsize=13)
                ax.set_xlabel('Contract')

                # --- Legend inside each subplot ---
                ax.legend(loc='upper right', fontsize=9, title='Gender & Churn', title_fontsize=10)

            axes[0].set_ylabel('Customer Count')

            # Adjust spacing slightly
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.2)

            fig.suptitle('Contract → Gender → Churn by SIM Provider', fontsize=16, y=0.98)

            plt.savefig(f'{Plot}/Contract_SIMProvider_Gender_Churn.png', bbox_inches='tight')
            plt.close()

            # --- Plot 9: Paperless Billing Vs Gender Vs Churn ---
            categories = df['PaperlessBilling'].unique()

            fig, axes = plt.subplots(1, len(categories), figsize=(13,8), sharey=True)
            axes = np.atleast_1d(axes)

            for ax, pb in zip(axes, categories):
                temp = df[df['PaperlessBilling'] == pb]

                # Crosstab counts
                counts = pd.crosstab(temp['gender'], temp['Churn'])

                x = np.arange(len(counts))
                width = 0.35

                # Plot bars
                ax.bar(x - width / 2, counts['No'], width, label='No Churn', color='teal')
                ax.bar(x + width / 2, counts['Yes'], width, label='Churn', color='orange')

                # Add values on top
                for i, (no, yes) in enumerate(zip(counts['No'], counts['Yes'])):
                    ax.text(x[i] - width / 2, no + 1, no, ha='center', va='bottom', fontsize=9)
                    ax.text(x[i] + width / 2, yes + 1, yes, ha='center', va='bottom', fontsize=9)

                ax.set_xticks(x)
                ax.set_xticklabels(counts.index)
                ax.set_title(f'PaperlessBilling = {pb}')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Customer Count')
                ax.legend()

            plt.suptitle('PaperlessBilling → Gender → Churn', fontsize=14)
            plt.tight_layout()

            plt.savefig(f'{Plot}/PaperlessBilling_Gender_Churn.png')
            plt.close()

            # --- Plot 10: PaymentMethod → Gender → Churn ---
            methods = df['PaymentMethod'].unique()

            fig, axes = plt.subplots(1, len(methods), figsize=(18,7), sharey=True)
            axes = np.atleast_1d(axes)

            for ax, method in zip(axes, methods):
                temp = df[df['PaymentMethod'] == method]

                counts = pd.crosstab(temp['gender'], temp['Churn'])
                x = np.arange(len(counts))
                width = 0.35

                ax.bar(x - width / 2, counts['No'], width, label='No Churn', color='teal')
                ax.bar(x + width / 2, counts['Yes'], width, label='Churn', color='orange')

                # Values on top
                for i, (no, yes) in enumerate(zip(counts['No'], counts['Yes'])):
                    ax.text(x[i] - width / 2, no + 1, no, ha='center', va='bottom', fontsize=9)
                    ax.text(x[i] + width / 2, yes + 1, yes, ha='center', va='bottom', fontsize=9)

                ax.set_xticks(x)
                ax.set_xticklabels(counts.index)
                ax.set_title(method)
                ax.set_xlabel('Gender')
                ax.set_ylabel('Customer Count')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', ncol=2, fontsize= 10)

            plt.suptitle('Payment Method → Gender → Churn', fontsize=14)
            plt.tight_layout()

            plt.savefig(f'{Plot}/Payment Method_Gender_Churn.png')
            plt.close()

            # --- Plot 11: Service_provider Vs Gender Vs Churn by Tenure ---
            bins = list(range(0, 76, 3))
            labels = [f'{b}-{b + 3}' for b in bins[:-1]]

            df['Tenure_Group'] = pd.cut(
                df['tenure'],
                bins=bins,
                labels=labels,
                right=False
            )

            # Unique providers & genders
            providers = sorted(df['Service_Provider'].unique())
            genders = sorted(df['gender'].unique())

            colors = {'No': 'green', 'Yes': 'red'}

            # Create subplot grid
            fig, axes = plt.subplots(
                nrows=len(providers),
                ncols=len(genders),
                figsize=(38, 35),
                sharex=True,
                sharey=False
            )

            plt.subplots_adjust(hspace=0.35, wspace=0.15)

            # Loop through providers & genders
            for row_idx, provider in enumerate(providers):
                for col_idx, gender in enumerate(genders):

                    ax = axes[row_idx, col_idx]

                    subset = df[
                        (df['Service_Provider'] == provider) &
                        (df['gender'] == gender)
                        ]

                    ct = (
                        pd.crosstab(subset['Tenure_Group'], subset['Churn'])
                        .reindex(labels, fill_value=0)
                    )

                    # Ensure both churn classes exist
                    for col in ['No', 'Yes']:
                        if col not in ct.columns:
                            ct[col] = 0

                    x = np.arange(len(labels))
                    width = 0.4

                    bars_no = ax.bar(
                        x - width / 2,
                        ct['No'],
                        width,
                        label='No Churn',
                        color=colors['No'],
                        edgecolor='black',
                        alpha=0.8
                    )

                    bars_yes = ax.bar(
                        x + width / 2,
                        ct['Yes'],
                        width,
                        label='Yes Churn',
                        color=colors['Yes'],
                        edgecolor='black',
                        alpha=0.8
                    )

                    # Title & labels
                    ax.set_title(
                        f'{provider} - {gender}',
                        fontsize=14,
                        fontweight='bold'
                    )
                    ax.set_ylabel('Count')

                    # Legend for EACH subplot
                    ax.legend(
                        loc='upper right',
                        fontsize=9,
                        frameon=True
                    )

                    # Add value labels
                    def autolabel(bars):
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                ax.annotate(
                                    f'{int(height)}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center',
                                    va='bottom',
                                    fontsize=9
                                )

                    autolabel(bars_no)
                    autolabel(bars_yes)

            # X-axis labels only for bottom row
            for ax in axes[-1, :]:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=11)
                ax.set_xlabel('Tenure Group (Months)', fontsize=14)

            # Super title
            plt.suptitle(
                'Churn Count by Tenure: Service Provider & Gender Analysis',
                fontsize=22,
                fontweight='bold',
                y=0.92
            )

            # Save plot
            plt.savefig(
                f'{Plot}/Tenure_Gender_churn.png',
                bbox_inches='tight',
                dpi=300
            )

            plt.close()



            # --- Plot 12: Monthly Charges (Intervals of 5) per Service Provider ---

            # 1. Create Bins for Monthly Charges
            # Range appears to be 15 to 120+, with step of 5
            bins = list(range(15, 126, 5))
            labels = [f'{b}-{b + 5}' for b in bins[:-1]]

            df['MonthlyCharges_Group'] = pd.cut(
                df['MonthlyCharges'],
                bins=bins,
                labels=labels,
                right=False
            )

            # 2. Setup Subplots (4 Rows, 1 Column)
            providers = ['Jio', 'BSNL', 'Vi', 'Airtel']  # Order based on your image
            fig, axes = plt.subplots(len(providers), 1, figsize=(18, 24))

            # Adjust space between rows
            plt.subplots_adjust(hspace=0.4)

            # 3. Loop through each provider and plot
            for i, provider in enumerate(providers):
                ax = axes[i]

                # Filter data for current provider
                subset = df[df['Service_Provider'] == provider]

                # Crosstab: Groups vs Churn
                # .reindex(labels) ensures all bins show up even if empty
                ct = (pd.crosstab(subset['MonthlyCharges_Group'], subset['Churn'])
                      .reindex(labels, fill_value=0))

                # Ensure 'No' and 'Yes' columns exist
                for col in ['No', 'Yes']:
                    if col not in ct.columns:
                        ct[col] = 0

                # Bar settings
                x_index = np.arange(len(labels))
                width = 0.35

                # Plot Bars
                # Using 'tab:green' and 'tab:red' to match the image style
                rects1 = ax.bar(x_index - width / 2, ct['No'], width, label='No Churn',
                                color='tab:green', edgecolor='black', alpha=0.9)
                rects2 = ax.bar(x_index + width / 2, ct['Yes'], width, label='Yes Churn',
                                color='tab:red', edgecolor='black', alpha=0.9)

                # Formatting the Subplot
                ax.set_title(f'Service Provider: {provider}', fontsize=18, fontweight='bold')
                ax.set_ylabel('Customer Count', fontsize=12)
                ax.set_xlabel('Monthly Charges ($)', fontsize=12)

                # X-Axis Ticks
                ax.set_xticks(x_index)
                ax.set_xticklabels(labels, rotation=45)

                # Grid and Legend
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.legend(loc='upper right')

                # 4. Add Value Labels (Vertical Text)
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        if height > 0:
                            ax.text(
                                rect.get_x() + rect.get_width() / 2,
                                height + 3,  # slightly above bar
                                f'{int(height)}',
                                ha='center',
                                va='bottom',
                                rotation=0,  # Vertical text like in the image
                                fontsize=10
                            )

                autolabel(rects1)
                autolabel(rects2)

                # Set Y limit slightly higher to fit the vertical text
                ax.set_ylim(0, max(ct['No'].max(), ct['Yes'].max()) * 1.15)

            # Overall Title
            plt.suptitle('Churn Analysis by Monthly Charges (Intervals of 5) per Service Provider',
                         fontsize=22, y=0.91)

            plt.savefig(f'{Plot}/MonthlyCharges_ServiceProvider_Churn.png', bbox_inches='tight')
            plt.close()

            # --- Plot 13: Region → Service Provider → Churn ---
            regions = df['Region_Type'].unique()
            providers = df['Service_Provider'].unique()
            fig, axes = plt.subplots(len(regions), 1, figsize=(15, 5 * len(regions)))

            # Add space between the charts
            plt.subplots_adjust(hspace=0.4)

            # Ensure 'axes' is a list even if there is only 1 region
            if len(regions) == 1:
                axes = [axes]

            for i, region in enumerate(regions):

                # Select the specific subplot axis
                ax = axes[i]

                # Filter data for the current Region
                subset = df[df['Region_Type'] == region]

                # Calculate counts of Churn (Yes/No) for each Service Provider
                counts = pd.crosstab(subset['Service_Provider'], subset['Churn']).reindex(providers, fill_value=0)

                # Safety check: ensure 'No' and 'Yes' columns exist
                if 'No' not in counts.columns: counts['No'] = 0
                if 'Yes' not in counts.columns: counts['Yes'] = 0

                # Define X positions
                x_pos = np.arange(len(providers))
                width = 0.35

                # Plot the Bars (Side-by-Side)
                rects1 = ax.bar(x_pos - width / 2, counts['No'], width, label='No Churn', color='#2ca02c',
                                edgecolor='black')
                rects2 = ax.bar(x_pos + width / 2, counts['Yes'], width, label='Yes Churn', color='#d62728',
                                edgecolor='black')


                ax.set_title(f'Region: {region}', fontsize=20, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(providers, fontsize=12)
                ax.set_xlabel('Service Provider', fontsize=14)
                ax.set_ylabel('Customer Count', fontsize=14)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.legend(loc='upper right', fontsize=12)

                # Add Numbers on Top of Bars
                for rect in rects1 + rects2:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f'{int(height)}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 5), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.suptitle('Churn Analysis: Region vs Service Provider', fontsize=24, y=0.98)
            plt.savefig(f'{Plot}/Region_ServiceProvider_Churn.png', bbox_inches='tight')
            plt.close()


            # --- Plot 14 : Region vs Service Provider vs Gender/Senior vs Churn ---

            regions = sorted(df['Region_Type'].unique())
            providers = sorted(df['Service_Provider'].unique())

            # Create a Grid of Subplots
            # Rows = Regions, Columns = Service Providers
            fig, axes = plt.subplots(
                nrows=len(regions),
                ncols=len(providers),
                figsize=(5 * len(providers), 6 * len(regions)),  # Adjust size dynamically
                sharey=True  # Share Y-axis for easier comparison across a row
            )

            # Adjust layout spacing
            plt.subplots_adjust(hspace=0.4, wspace=0.1)

            # Loop through Regions (Rows) and Providers (Columns)
            for i, region in enumerate(regions):
                for j, provider in enumerate(providers):

                    # Handle 1D array of axes if only 1 region or 1 provider exists
                    if len(regions) == 1 and len(providers) == 1:
                        ax = axes
                    elif len(regions) == 1:
                        ax = axes[j]
                    elif len(providers) == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i, j]

                    # 1. Filter Data for specific Region AND Service Provider
                    subset = df[
                        (df['Region_Type'] == region) &
                        (df['Service_Provider'] == provider)
                        ]

                    # 2. Group by Demographics (Gender + Senior) and Churn
                    # We group by [Gender, Senior] -> Count of Churn
                    data = (subset.groupby(['gender', 'SeniorCitizen', 'Churn'])
                            .size()
                            .unstack(fill_value=0))

                    # Ensure 'No' and 'Yes' columns exist
                    for col in ['No', 'Yes']:
                        if col not in data.columns:
                            data[col] = 0

                    # 3. Plotting
                    x = np.arange(len(data))
                    width = 0.35

                    # Green Bars (No Churn)
                    rects1 = ax.bar(x - width / 2, data['No'], width, label='No Churn',
                                    color='#2ca02c', edgecolor='black', alpha=0.9)
                    # Red Bars (Yes Churn)
                    rects2 = ax.bar(x + width / 2, data['Yes'], width, label='Yes Churn',
                                    color='#d62728', edgecolor='black', alpha=0.9)

                    # 4. Labels & Titles
                    # Only set Title for the top row to reduce clutter (optional)
                    if i == 0:
                        ax.set_title(f'{provider}', fontsize=18, fontweight='bold', pad=20)

                    # Set Row labels (Region) on the left-most charts
                    if j == 0:
                        ax.set_ylabel(f'{region}\nCustomer Count', fontsize=14, fontweight='bold')

                    # X-Axis Labels
                    labels = [f'{g}\nSr={s}' for g, s in data.index]
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=0, fontsize=9)

                    ax.grid(axis='y', linestyle='--', alpha=0.3)

                    # Legend only in the top-right plot to save space
                    if i == 0 and j == len(providers) - 1:
                        ax.legend(title='Churn', loc='upper right')

                    # 5. Add Value Annotations
                    for rect in rects1 + rects2:
                        height = rect.get_height()
                        if height > 0:
                            ax.annotate(f'{int(height)}',
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=8)

            # Global Title
            plt.suptitle('Churn Analysis: Region vs Service Provider vs Demographics', fontsize=24, y=0.95)

            # Save
            plt.savefig(f'{Plot}/Region_Provider_Demographics_Churn.png', bbox_inches='tight')
            plt.close()

            # --- Plot 15 :  Internet Service which Service Provider is Dominance ---

            # Filter to focus on actual internet users (DSL & Fiber)
            internet_data = df[df['InternetService'].isin(['DSL', 'Fiber optic', 'No'])]

            # Crosstab to count users per provider per technology
            market_share = pd.crosstab(internet_data['InternetService'], internet_data['Service_Provider'])

            # Plot Grouped Bar Chart
            ax = market_share.plot(kind='bar', figsize=(10, 7), width=0.8, edgecolor='black')

            plt.title('Internet Serives with Service Provider', fontsize=16, fontweight='bold')
            plt.xlabel('Internet Type', fontsize=12)
            plt.ylabel('Customer Count', fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Service Provider')
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # Add counts on top of bars
            for container in ax.containers:
                ax.bar_label(container, padding=3, fontsize=10)

            plt.tight_layout()
            plt.savefig(f'{Plot}/InternetService with ServiceProvider.png')
            plt.close()


            # --- Plot 16 : Plot All Services Vs Churn ---

            service_cols = [
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies'
            ]

            plt.figure(figsize=(18, 10))

            # Loop through each column and create subplot
            for i, col in enumerate(service_cols, 1):
                plt.subplot(2, 3, i)

                # Calculate percentage
                pct = df[col].value_counts(normalize=True) * 100

                ax = pct.plot(kind='bar', color=['red', 'green', 'gold'])

                plt.title(f'{col} Usage (%)')
                plt.xlabel('',rotation=0)
                plt.ylabel('Percentage')

                # Add percentage labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%')

            plt.tight_layout()
            plt.savefig(f'{Plot}/All Services Vs Churn.png')
            plt.close()


















        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')