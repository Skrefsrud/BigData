# Oppgave 1: Top 10 land som bidrar til størst grad av overnattinger

import pandas as pd
import matplotlib.pyplot as plt

# Laste dataen
file_path = 'H1.csv'
df = pd.read_csv(file_path)

# Grupper dataen etter 'Country' og teller antall unike tilfeller
# begrenser til de 10 første
top_10_countries = df['Country'].value_counts().head(10)
print(top_10_countries)


# Oppagve 2: filtrere ut kanselerete bookinger og gruppere etter markeds segment

# Filtrere ut kanselerte bookinger
df_not_canceled = df[df['IsCanceled'] == 0]

# Grupper etter markedssegment og summerer inntjening
segment_earning = df_not_canceled.groupby(
    'MarketSegment')['ADR'].sum().reset_index()

segment_earning.columns = ['Market segment', 'Total Earnings']

print('Total earning by market segment excluding canceled bookings: ')
print(segment_earning)


# Oppgave 3: Historgram for prisen på ADR
# Figure size
plt.figure(figsize=(10, 6))

# Set data, bins and range for histogram
plt.hist(df['ADR'], bins=30, range=(
    df['ADR'].min(), df['ADR'].max()), edgecolor='black')


# Labels and title
plt.title('Prices for rooms', fontsize=14)
plt.xlabel('Average daily rate (ADR)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Shot plot
plt.show()


# Oppgave 4: Linjediagram for gjennomsnitt rompris og antall kansellasjoner pr måned

# Filter for år 2016
df_2016 = df[df['ArrivalDateYear'] == 2016]

# Grupper basert på måned, kalkuler gjennomsnittlig ADR og antall kansellasjoner

monthly_data = df_2016.groupby('ArrivalDateMonth').agg(
    average_adr=('ADR', 'mean'),
    cancellations=('IsCanceled', sum)
).reset_index()

# Plot linjediagrammet

plt.figure(figsize=(10, 6))

# Gjennomsnitllig ADR
plt.plot(monthly_data['ArrivalDateMonth'],
         monthly_data['average_adr'], label='Average ADR', marker='o')

# Antall kansellasjoner
plt.plot(monthly_data['ArrivalDateMonth'], monthly_data['cancellations'],
         label='Cancellations', marker='o', color='red')


# Tittel og labels
plt.title('Average ADR and Number of Cancellations by month in 2016', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Legends for å differensiere mellom linjene
plt.legend()

# x-akse ticks
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr',
           'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


plt.show()
