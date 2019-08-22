# Bank Marketing

Ashley O'Mahony | [ashleyomahony.com](http://ashleyomahony.com) | June 2019  

***

This project analyses a **bank marketing campaign history** and predicts **which customers are likely to subscribe to a bank term deposit**. The original dataset can be found on [Kaggle](https://www.kaggle.com/henriqueyamahata/bank-marketing).  

All the files of this project are saved in a [GitHub repository](https://github.com/ashomah/Bank-Marketing).  

<br>

#### Large Files  

To comply with GitHub file size limits, the folders containing large files (data, models...) have been compressed and split by parts of 100 MB, and stored on Google Drive (see links in relative folders). To decompress these files, download all parts and start decompressing the first file (ending with `.001`).  

<br>

#### Report Formats  

The report of this analysis comes in three formats:  

* **GitHub Markdown**: to be easily read on the [GitHub repository](https://github.com/ashomah/Bank-Marketing).  
* **HTML**: to be consulted in a web browser and/or printed.  
* **Shiny App**: to add interactivity and provide enhanced readability.  

<br>

#### For developers  

To visualize the Rmarkdown in your web browser and see any update on saving, `cd` to the directory and use this command in Terminal:  
`Rscript -e 'rmarkdown::run("Bank-Marketing-Report.Rmd", shiny_args = list(launch.browser = TRUE))'`
