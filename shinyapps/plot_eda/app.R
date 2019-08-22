###
### THIS APP ALLOW TO DISPLAY RELEVANT PLOTS FOR THE DATASET
###


# Load Packages
library('ggthemes')
library('ggplot2')
library('plyr')
library('grid')
library('gridExtra')
library('shiny')
library('shinyjs')

# Load Data
df <- readRDS('data/bank_train.rds')

for (i in c('age', 'balance', 'day', 'duration', 'pdays')){
    df[, i] <- as.numeric(df[ ,i])
}

for (i in c('job', 'marital', 'education', 'previous', 'campaign', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y')){
    df[, i] <- as.factor(df[ ,i])
}


# Define UI for application
ui <- fluidPage(
    wellPanel(
        selectInput(
            inputId = 'feature',
            label = 'Feature',
            choices = sort(names(df)),
            selected = 'age'
        ),
        textOutput('class_feat')
    ),
    
    conditionalPanel("output.class_feat == 'factor'",
                     plotOutput("fact_plot_1", height = 200)),
    conditionalPanel("output.class_feat == 'factor'",
                     plotOutput("fact_plot_2", height = 200)),
    conditionalPanel("output.class_feat != 'factor'",
                     plotOutput("num_plot_1", height = 200)),
    conditionalPanel("output.class_feat != 'factor'",
                     plotOutput("num_plot_2", height = 200)),
    conditionalPanel("output.class_feat != 'factor'",
                     plotOutput("num_plot_3", height = 200))
)

# Define server logic
server <- function(input, output) {
    
    output$class_feat <- reactive({
        class_feat <- class(df[, input$feature])
    })
    
    
    output$fact_plot_1 <- renderPlot({
        if (class(df[, input$feature]) == 'factor'){
            ggplot(data = df, aes(x = df[, input$feature])) +
            geom_bar(color = 'darkcyan', fill = 'darkcyan', alpha = 0.4) +
            theme(axis.text.x=element_text(size=10, angle=90,hjust=0.95,vjust=0.2))+
            xlab(df[, input$feature])+
            ylab("Percent")+
            theme_tufte(base_size = 5, ticks=F)+
            theme(plot.margin = unit(c(10,10,10,10),'pt'),
                  axis.title=element_blank(),
                  axis.text = element_text(size = 10, family = 'Helvetica'),
                  axis.text.x = element_text(hjust = 1, size = 10, family = 'Helvetica', angle = 45),
                  legend.position = 'None')
        }
    })
    
    output$fact_plot_2 <- renderPlot({
        if (class(df[, input$feature]) == 'factor'){
            mytable <- table(df[, input$feature], df$y)
            tab <- as.data.frame(prop.table(mytable, 2))
            colnames(tab) <-  c('var', "y", "perc")

        ggplot(data = tab, aes(x = var, y = perc)) +
            geom_bar(aes(fill = y),stat = 'identity', position = 'dodge', alpha = 2/3) +
            theme(axis.text.x=element_text(size=10, angle=90,hjust=0.95,vjust=0.2))+
            xlab(df[, input$feature])+
            ylab("Percent")+
            theme_tufte(base_size = 5, ticks=F)+
            theme(plot.margin = unit(c(10,10,10,10),'pt'),
                  axis.title=element_blank(),
                  axis.text = element_text(size = 10, family = 'Helvetica'),
                  axis.text.x = element_text(hjust = 1, size = 10, family = 'Helvetica', angle = 45),
                  legend.position = 'None')
        }
    })
    
    output$num_plot_1 <- renderPlot({
        if (class(df[, input$feature]) != 'factor'){
            ggplot(df,
               aes(x = df[, input$feature])) +
            geom_density(color = 'darkcyan', fill = 'darkcyan', alpha = 0.4) +
            geom_vline(aes(xintercept=median(df[, input$feature])),
                       color="darkcyan", linetype="dashed", size=1) +
            theme_minimal() +
            theme(panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank())+
            labs(x = paste0(toupper(substr(input$feature, 1, 1)), tolower(substr(
                input$feature, 2, nchar(input$feature)))),
                y = 'Density')
        }
    })
    
    output$num_plot_2 <- renderPlot({
        if (class(df[, input$feature]) != 'factor'){
            ggplot(df, aes(y=df[, input$feature])) +
            geom_boxplot(fill = "darkcyan", color = 'darkcyan', outlier.colour = 'darkcyan', alpha=0.4)+
            coord_flip()+
            theme_tufte(base_size = 5, ticks=F)+
            theme(plot.margin = unit(c(10,10,10,10),'pt'),
                  axis.title=element_blank(),
                  axis.text = element_text(size = 10, family = 'Helvetica'),
                  axis.text.x = element_text(hjust = 1, size = 10, family = 'Helvetica'),
                  legend.position = 'None')    +
                labs(x = paste0(toupper(substr(input$feature, 1, 1)), tolower(substr(
                    input$feature, 2, nchar(input$feature)))))
            
        }
        })
    
    output$num_plot_3 <- renderPlot({
        if (class(df[, input$feature]) != 'factor'){
            # mu <- ddply(df, "y", summarise, grp.median=median(df[, input$feature]))
        ggplot(data = df, aes(x = df[, input$feature], color = y, group = y, fill = y)) + 
            geom_density(alpha=0.4) +
            # geom_vline(data=df, aes(xintercept=median(df[df$y == 1, input$feature])), linetype = "dashed", size = 1)+
            theme_minimal()+
            theme(panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank(), 
                  panel.border = element_blank()) + 
            labs(x = paste0(toupper(substr(input$feature, 1, 1)), tolower(substr(
                input$feature, 2, nchar(input$feature)))),
                y = 'Density')
        }
        })
    
}

# Run the application 
shinyApp(ui = ui, server = server, options = list(height = 800))
