from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
import tempfile
import os
from datetime import datetime

def generate_yield_report(prediction_data):
    """
    Generate a PDF report for a yield prediction
    """
    # Create a temporary file for the report
    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, f"yield_report_{prediction_data['id']}.pdf")
    
    # Create the PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center aligned
    )
    
    story.append(Paragraph("Agricultural Yield Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report generated on: {report_date}", meta_style))
    story.append(Paragraph(f"Prediction ID: {prediction_data['id']}", meta_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Farmer and farm details
    story.append(Paragraph("Farm Details", styles['Heading2']))
    
    farm_data = [
        ['Farmer Name:', f"{prediction_data.get('first_name', '')} {prediction_data.get('last_name', '')}"],
        ['Farm Name:', prediction_data.get('farm_name', 'N/A')],
        ['Location:', prediction_data.get('location', 'N/A')],
        ['Crop Type:', prediction_data.get('crop_type', 'N/A')]
    ]
    
    farm_table = Table(farm_data, colWidths=[1.5*inch, 3*inch])
    farm_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))
    
    story.append(farm_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Input parameters
    story.append(Paragraph("Input Parameters", styles['Heading2']))
    
    input_data = [
        ['Parameter', 'Value'],
        ['Temperature (°C)', f"{prediction_data.get('temperature_avg', 'N/A')}"],
        ['Rainfall (mm)', f"{prediction_data.get('rainfall_total', 'N/A')}"],
        ['Humidity (%)', f"{prediction_data.get('humidity_avg', 'N/A')}"],
        ['Soil pH', f"{prediction_data.get('soil_ph', 'N/A')}"],
        ['Fertilizer (kg/hectare)', f"{prediction_data.get('fertilizer_planned', 'N/A')}"],
        ['Irrigation (mm)', f"{prediction_data.get('irrigation_planned', 'N/A')}"]
    ]
    
    input_table = Table(input_data, colWidths=[2*inch, 1.5*inch])
    input_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(input_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Prediction results
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    
    result_data = [
        ['Predicted Yield', f"{prediction_data.get('predicted_yield', 'N/A'):.2f} tons/hectare"],
        ['Confidence Level', f"{prediction_data.get('confidence', 0) * 100:.1f}%"],
        ['Model Used', prediction_data.get('model_used', 'N/A')],
        ['Prediction Date', prediction_data.get('created_at', 'N/A')]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgreen),
    ]))
    
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations section
    story.append(Paragraph("Recommendations", styles['Heading2']))
    
    # Generate recommendations based on prediction
    yield_value = prediction_data.get('predicted_yield', 0)
    crop_type = prediction_data.get('crop_type', '').lower()
    
    recommendations = []
    
    if crop_type == 'wheat':
        if yield_value < 3.0:
            recommendations.append("Consider increasing fertilizer application by 10-15%")
            recommendations.append("Ensure adequate irrigation during critical growth stages")
            recommendations.append("Test soil nutrients and amend based on results")
        elif yield_value > 4.5:
            recommendations.append("Your practices are optimal. Maintain current approach")
            recommendations.append("Consider crop rotation to maintain soil health")
    elif crop_type == 'corn':
        if yield_value < 5.0:
            recommendations.append("Consider increasing nitrogen application")
            recommendations.append("Ensure proper plant spacing for optimal growth")
            recommendations.append("Monitor for pests and diseases regularly")
        elif yield_value > 7.0:
            recommendations.append("Excellent yield expected. Consider expanding production")
    else:
        recommendations.append("Monitor crop health regularly and adjust practices as needed")
        recommendations.append("Consider soil testing to optimize nutrient management")
    
    if not recommendations:
        recommendations.append("No specific recommendations available for this crop and yield combination")
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    
    return report_path