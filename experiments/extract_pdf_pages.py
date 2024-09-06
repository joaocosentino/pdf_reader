from PyPDF2 import PdfReader, PdfWriter

def pdf_slice(pdf_path, first_page, last_page ):
    '''
    Cria um recorte de PDF a partir do PDF em pdf_path, da página first_page até
    a página last_page.
    '''
    # Carrega o PDF
    leitor_pdf = PdfReader(pdf_path)
    escritor_pdf = PdfWriter()

    # Seleciona as páginas que você quer extrair (ex: páginas 1 e 2)
    paginas_para_extrair = list(range(first_page, last_page))  # Índices começam em 0

    for pagina in paginas_para_extrair:
        escritor_pdf.add_page(leitor_pdf.pages[pagina])

    # Salva o novo PDF
    with open(f'./pdf_files/owner_manual_p{first_page}-p{last_page}.pdf', 'wb') as novo_pdf:
        escritor_pdf.write(novo_pdf)

    print("Novo PDF criado com sucesso!")


pdf_slice('./pdf_files/Owners_Manual-Ram_1500_25_Crew_Cab.pdf', 283, 300)