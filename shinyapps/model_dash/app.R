###
### THIS APP ALLOW TO DISPLAY A DASHBOARD FOR MODELS
###


# Load Packages
library('shiny')
library('shinydashboard')
library('ggplot2')
library('caret')
library('pROC')
library('ggthemes')
library('corrplot')
library('glmnet')
library('MLmetrics')
library('ranger')
library('xgboost')
library('factoextra')
library('kableExtra')
library('RColorBrewer')
library('tufte')
library('flexclust')
library('factoextra')
library('AUC')


# Load Data
results <- readRDS('data/all_real_results.rds')
file_list <- readRDS('data/file_list.rds')

for (m in seq(nrow(file_list))){
    # assign(paste0(file_list[m, 'model_file']), readRDS(file = paste0('data/', file_list[m, 'model_file'], '.rds')))
    assign(paste0(file_list[m, 'cm_file']), readRDS(file = paste0('data/', file_list[m, 'cm_file'], '.rds')))
    assign(paste0(file_list[m, 'roc']), readRDS(file = paste0('data/', file_list[m, 'roc'], '.rds')))
    assign(paste0(file_list[m, 'density']), readRDS(file = paste0('data/', file_list[m, 'density'], '.rds')))
}



# Define UI for application
ui <- dashboardPage(skin = 'black',
                    dashboardHeader(title = "Models Dashboard"),
                    dashboardSidebar(
                        selectInput(
                            inputId = 'model',
                            label = 'Model',
                            choices = rownames(results),
                            selected = 'Ranger FE2 Binning'
                        )
                    ),
                    dashboardBody(
                        
                        fluidRow(
                            valueBoxOutput("sensibility"),
                            valueBoxOutput("accuracy"),
                            valueBoxOutput("precision"),
                            valueBoxOutput("recall"),
                            valueBoxOutput("f1"),
                            valueBoxOutput("coef")
                        ),
                        
                        fluidRow(
                            box(title = 'Confusion Matrix',
                                status = 'info',
                                plotOutput('conf_mat', height = 250)),
                            
                            box(title = 'ROC Curve',
                                status = 'info',
                                plotOutput('roc', height = 250))
                        ),
                        fluidRow(
                            box(title = 'Density Plot',
                                status = 'info',
                                plotOutput('dens_plot')
                                , width = 12
                            )
                        )
                    )
)

# Define server logic
server <- function(input, output) {
    
    output$model_file <- reactive({
        results[rownames(results) == input$model, 'File']
    })
    
    output$sensibility <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'Sensitivity'], 5),
            subtitle = "Sensitivity",
            color = 'yellow',
            icon = icon("sad-cry")
        )
    })
    
    output$accuracy <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'Accuracy'], 5),
            subtitle = "Accuracy",
            color = 'teal',
            icon = icon("expand")
        )
    })
    
    output$precision <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'Precision'], 5),
            subtitle = "Precision",
            color = 'teal',
            icon = icon("crosshairs")
        )
    })
    
    output$recall <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'Specificity'], 5),
            subtitle = "Specificity",
            color = 'teal',
            icon = icon("angle-double-left")
        )
    })
    
    output$f1 <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'F1 Score'], 5),
            subtitle = "F1 Score",
            color = 'teal',
            icon = icon("flag-checkered")
        )
    })
    
    output$coef <- renderValueBox({
        valueBox(
            value = round(results[rownames(results) == input$model, 'Coefficients'], 5),
            subtitle = "Coefficients",
            color = 'teal',
            icon = icon("calculator")
        )
    })
    
    output$conf_mat <- renderPlot({
        data <- get(paste0(file_list[rownames(file_list) == input$model, 'cm_file']))
        fourfoldplot(data$table)
    })
    
    output$roc <- renderPlot({
        data <- get(paste0(file_list[rownames(file_list) == input$model, 'roc']))
        plot(data , lwd=4)
    })
    
    output$dens_plot <- renderPlot({
        data <- get(paste0(file_list[rownames(file_list) == input$model, 'density']))
        data
    })
    
}

# Run the application 
shinyApp(ui = ui, server = server, options = list(height = 500))
