# Tutorial to do automated literature search in Colab

### [Colab notebook :Automated Literature Retrieval/Search](https://colab.research.google.com/drive/1Fg9DmgES0GroMmeB9_cBfCcDiR8uFSVH)

#### This colab notebook describes the automated literature search on any query term and analyzing the results with the use of semanticClimate tools (`pygetpapers`, `amilib` and `docanalysis`)

- **[pygetpapers](https://github.com/petermr/pygetpapers)** developed by **Ayush Garg**
- **[docanalysis](https://github.com/petermr/docanalysis)** developed by **Shweata N Hegde**
- **[amilib](https://github.com/petermr/amilib)** developed by **Prof.Peter Murray-Rust**

### **Step 1: Installation of semanticClimate toolkit ```pygetpapers```, ```amilib``` and ```docanalysis```**

- ### pygetpapers : literature search

- ### amilib : make JQuery Datatable

- ### docanalysis: entity search on the scientific literature corpus

```
!pip install --quiet pygetpapers
!pip install --quiet amilib==0.3.9
!pip install --quiet docanalysis
```

### **Step 2 : use of tool `pygetpapers` to search open published literatures on the query.**

**The following code will give the number of articles published on the query term**.

**Query is provided either single word or multiple words joined by AND.**

```
!pygetpapers --query '"phytochemicals" AND "Invasive plant"' --xml -n --output phytochem --save_query
```

### **Step 3: To get the number of articles for specific time frame**.

```
!pygetpapers --query '"phytochemicals" AND "Invasive plant"' --xml -n --startdate "2010-01-01" --enddate "2024-10-31" --output phytochem --save_query
```

### **Step 4: This step will download metadata for 100 articles and saved in `phytochem`. The pdf can be downloaded with addition `-p` in the code.**

```
!pygetpapers --query '"phytochemicals" AND "Invasive plant"' --xml --limit 100 --output phytochem --save_query
```

### **Step 5: To create JQuery Datatables for the retrieved articles**.

- This give the summary of all the retrieved articles for easy navigation.

```
!amilib HTML --operation DATATABLES --indir phytochem
```

### **Step 6: Displaying Datatables (HTML output) in Colab environment**

```
from IPython.core.display import display, HTML

# Path to the HTML file
html_file_path = '/content/phytochem/datatables.html'

# Read the HTML file
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Display the HTML content
display(HTML(html_content))
```

### **Step 7: To extract list of Countries mentioned in the literature. the search is limited to specific section like Introduction (INT), Result (RES), Conclusion (CON) and Discussion (DIS) of the articles**.

```
!docanalysis --project_name phytochem --make_section --search_section INT, RES, CON, DIS --dictionary COUNTRY --output phyto_invasive.csv
```

### **Step 8: Displaying wordcloud for country list made from csv (output) in Colab environment**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/phyto_invasive.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_country.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)
```

### **Step 9: The list of plant mentioned in the scientific literature corpus is obtained by running below code.**

- the tool `docanalysis` performs the entity search using dictionary.

```
!docanalysis --project_name phytochem --make_section --dictionary EO_PLANT --output phytoinvasive.csv --make_json phytoinvasive.json
```

### **Step 10: To Display the wordcloud for all the plants present in the retrieved articles.**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/phytoinvasive.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_plant.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)

```

### **Step 11: The list of chemical compounds mentioned in the scientific literature corpus is obtained by running below code.**

```
!docanalysis --project_name phytochem --make_section --search_section INT, RES, CON, DIS --dictionary EO_COMPOUND --output comp_invasive.csv --make_json compinvasive.json
```

### **Step 12: Displaying list of compounds in a wordcloud.**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/comp_invasive.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_plant.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)

```

### **Step 13: To search the name of Drugs occuring in the retrieved articles.**

```
!docanalysis --project_name phytochem --make_section --search_section INT, RES, CON, DIS --dictionary DRUG --output drug_invasive.csv --make_json druginvasive.json
```

### **Step 14: Display the list of drugs in colab environment.**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/drug_invasive.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_plant.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)
```

### **Step 15: Extract the diseases discussed in the retrieved articles.**

```
!docanalysis --project_name phytochem --make_section --search_section CON, DIS --dictionary DISEASE --output disease_all1.csv --make_json dis_invasive.json
```

### **Step 16: Displaying the diseases mentioned in the retrieved literatures.**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/disease_all1.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_plant.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)

```

### **Step 17: Extract the target organisms discussed in the retrieved articles.**

```
!docanalysis --project_name phytochem --make_section --search_section CON, DIS --dictionary EO_TARGET --output target_info.csv --make_json targetlist_invasive.json
```

### **Step 18: Displaying the target organisms mentioned in the retrieved literatures.**

```
# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file (replace 'your_file.csv' with your actual file path)
file_path = "/content/phytochem/target_info.csv"  # Update this to the actual file path
data = pd.read_csv(file_path)

# Extract the column data
column_data = data['0']  # Replace '0' with the actual column name if different

# Split and count occurrences of each plant
from collections import Counter

# Split the column values by commas and count occurrences
plant_counter = Counter()
for entry in column_data.dropna():  # Remove any NaN values
    plants = [plant.strip() for plant in entry.split(',')]
    plant_counter.update(plants)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(plant_counter)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No axis
plt.show()

# Save the word cloud as an image file
# wordcloud_file_path = '/content/list_plant.png'
# wordcloud.to_file(wordcloud_file_path)

# Download the word cloud image if in Colab
# if files:
#     files.download(wordcloud_file_path)

```

### References:

- **[pygetpapers](https://github.com/petermr/pygetpapers)** developed by **Ayush Garg**
- **[docanalysis](https://github.com/petermr/docanalysis)** developed by **Shweata N Hegde**
- **[amilib](https://github.com/petermr/amilib)** developed by **Prof.Peter Murray-Rust**
