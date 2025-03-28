from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables to securely manage database credentials
load_dotenv()

# Database connection parameters retrieved from environment variables
URI = os.getenv("URI")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Prefix for CSV file paths to enable local file loading in Neo4j
CSV_PATH = "file:///"

# Queries to create constraints for each node type
CONSTRAINT_QUERIES = {
    "Adjuvant": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Adjuvant) REQUIRE n.c_adjuvant_id IS UNIQUE;
    """,
    "Gene": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Gene) REQUIRE n.c_gene_id IS UNIQUE;
    """,
    "GeneEngineering": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeneEngineering) REQUIRE n.c_gene_engineering_id IS UNIQUE;
    """,
    "Host": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Host) REQUIRE n.c_host_id IS UNIQUE;
    """,
    "HostGeneResponse": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:HostGeneResponse) REQUIRE n.c_host_gene_response_id IS UNIQUE;
    """,
    "HostResponse": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:HostResponse) REQUIRE n.c_host_response_id IS UNIQUE;
    """,
    "Pathogen": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Pathogen) REQUIRE n.c_pathogen_id IS UNIQUE;
    """,
    "Reference": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Reference) REQUIRE n.c_reference_id IS UNIQUE;
    """,
    "Vaccine": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vaccine) REQUIRE n.c_vaccine_id IS UNIQUE;
    """,
    "VaccineDetail": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:VaccineDetail) REQUIRE n.c_vaccine_detail_id IS UNIQUE;
    """,
    "Vaxjo": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vaxjo) REQUIRE n.c_vaxjo_id IS UNIQUE;
    """,
    "Vaxvec": """
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vaxvec) REQUIRE n.c_vaxvec_id IS UNIQUE;
    """,
}

# Queries to load data into Neo4j
CREATE_QUERIES = {
    "Adjuvant": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_adjuvant.csv' AS row
        MERGE (a:Adjuvant {{c_adjuvant_id: toInteger(row.c_adjuvant_id)}})
        SET a.c_adjuvant_vo_id = row.c_adjuvant_vo_id,
            a.c_adjuvant_label = row.c_adjuvant_label,
            a.c_adjuvant_description = row.c_adjuvant_description,
            a.c_vaccine_id = toInteger(row.c_vaccine_id),
            a.c_pathogen_id = toInteger(row.c_pathogen_id),
            a.c_curation_flag = row.c_curation_flag;
    """,
    
    "Gene": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_gene.csv' AS row
        MERGE (g:Gene {{c_gene_id: toInteger(row.c_gene_id)}})
        SET g.c_gene_name = row.c_gene_name,
            g.c_ncbi_gene_id = row.c_ncbi_gene_id,
            g.c_xrefs = row.c_xrefs,
            g.c_phi_annotation = row.c_phi_annotation,
            g.c_dna_sequence = row.c_dna_sequence,
            g.c_protein_sequence = row.c_protein_sequence,
            g.c_strain = row.c_strain,
            g.c_taxonomy_id = row.c_taxonomy_id,
            g.c_submitted_by = row.c_submitted_by,
            g.c_submission_date = row.c_submission_date,
            g.c_ncbi_protein_id = row.c_ncbi_protein_id,
            g.c_protein_refseq = row.c_protein_refseq,
            g.c_protein_name = row.c_protein_name,
            g.c_chromosome = row.c_chromosome,
            g.c_segment = row.c_segment,
            g.c_gene_locus_tag = row.c_gene_locus_tag,
            g.c_gene_start = row.c_gene_start,
            g.c_gene_end = row.c_gene_end,
            g.c_gene_strand = row.c_gene_strand,
            g.c_gene_refseq = row.c_gene_refseq,
            g.c_protein_note = row.c_protein_note,
            g.c_protein_annotation = row.c_protein_annotation,
            g.c_protein_pi = row.c_protein_pi,
            g.c_protein_weight = row.c_protein_weight,
            g.c_protein_length = row.c_protein_length,
            g.c_phi_function = row.c_phi_function,
            g.c_plasmid = row.c_plasmid,
            g.c_pathogen_id = toInteger(row.c_pathogen_id),
            g.c_curation_flag = row.c_curation_flag,
            g.c_ncbi_db = row.c_ncbi_db,
            g.c_ncbi_nucleotide_id = row.c_ncbi_nucleotide_id,
            g.c_phi_function2 = row.c_phi_function2,
            g.c_phi_annotation2 = row.c_phi_annotation2,
            g.c_tmp = row.c_tmp,
            g.c_cog = row.c_cog,
            g.c_cog_cat = row.c_cog_cat,
            g.c_vaxign_computed = row.c_vaxign_computed,
            g.c_vo_id = row.c_vo_id,
            g.c_pdb_id = row.c_pdb_id;
    """,
    
    "GeneEngineering": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_gene_engineering.csv' AS row
        MERGE (ge:GeneEngineering {{c_gene_engineering_id: toInteger(row.c_gene_engineering_id)}})
        SET ge.c_vaccine_id = toInteger(row.c_vaccine_id),
            ge.c_pathogen_id = toInteger(row.c_pathogen_id),
            ge.c_gene_id = toInteger(row.c_gene_id),
            ge.c_type = row.c_type,
            ge.c_description = row.c_description,
            ge.c_gene_engineering_name = row.c_gene_engineering_name,
            ge.c_curation_flag = row.c_curation_flag;
    """,
    
    "Host": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_host.csv' AS row
        MERGE (h:Host {{c_host_id: toInteger(row.c_host_id)}})
        SET h.c_host_name = row.c_host_name,
            h.c_scientific_name = row.c_scientific_name,
            h.c_taxonomy_id = row.c_taxonomy_id,
            h.c_submitted_by = row.c_submitted_by,
            h.c_curation_flag = row.c_curation_flag,
            h.c_pathogen_id = toInteger(row.c_pathogen_id);
    """,
    
    "HostGeneResponse": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_host_gene_response.csv' AS row
        MERGE (hgr:HostGeneResponse {{c_host_gene_response_id: toInteger(row.c_host_gene_response_id)}})
        SET hgr.c_pathogen_id = toInteger(row.c_pathogen_id),
            hgr.c_description = row.c_description,
            hgr.c_host_response_id = toInteger(row.c_host_response_id),
            hgr.c_gene_id = toInteger(row.c_gene_id),
            hgr.c_curation_flag = row.c_curation_flag,
            hgr.c_response_type_vo_id = row.c_response_type_vo_id,
            hgr.c_response_type_vo_id2 = row.c_response_type_vo_id2;
    """,
    
    "HostResponse": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_host_response.csv' AS row
        MERGE (hr:HostResponse {{c_host_response_id: toInteger(row.c_host_response_id)}})
        SET hr.c_host_id = toInteger(row.c_host_id),
            hr.c_vaccine_id = toInteger(row.c_vaccine_id),
            hr.c_immune_response = row.c_immune_response,
            hr.c_pathogen_id = toInteger(row.c_pathogen_id),
            hr.c_host_strain = row.c_host_strain,
            hr.c_vaccination_protocol = row.c_vaccination_protocol,
            hr.c_route = row.c_route,
            hr.c_persistence = row.c_persistence,
            hr.c_protection_efficacy = row.c_protection_efficacy,
            hr.c_side_effects = row.c_side_effects,
            hr.c_challenge_protocol = row.c_challenge_protocol,
            hr.c_description = row.c_description,
            hr.c_curation_flag = row.c_curation_flag,
            hr.c_deleted = row.c_deleted,
            hr.c_immune_response_assay_type = row.c_immune_response_assay_type,
            hr.c_immune_response_type = row.c_immune_response_type,
            hr.c_immune_response_type2 = row.c_immune_response_type2,
            hr.c_age = row.c_age,
            hr.c_gender = row.c_gender,
            hr.c_cellType = row.c_cellType;
    """,
    
    "Pathogen": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_pathogen.csv' AS row
        MERGE (p:Pathogen {{c_pathogen_id: toInteger(row.c_pathogen_id)}})
        SET p.c_pathogen_name = row.c_pathogen_name,
            p.c_taxon_id = row.c_taxon_id,
            p.c_pathogenesis = row.c_pathogenesis,
            p.c_disease_name = row.c_disease_name,
            p.c_protective_immunity = row.c_protective_immunity,
            p.c_submitted_by = row.c_submitted_by,
            p.c_submission_date = row.c_submission_date,
            p.c_curation_flag = row.c_curation_flag,
            p.c_reviewer = row.c_reviewer,
            p.c_host_range = row.c_host_range,
            p.c_full_text = row.c_full_text,
            p.c_introduction = row.c_introduction,
            p.c_gram = row.c_gram,
            p.c_organism_type = row.c_organism_type,
            p.c_preparation_vo_id = row.c_preparation_vo_id,
            p.c_vaccine_vo_id = row.c_vaccine_vo_id,
            p.c_protein_vo_id = row.c_protein_vo_id;
    """,
    
    "Reference": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_reference.csv' AS row
        MERGE (r:Reference {{c_reference_id: toInteger(row.c_reference_id)}})
        SET r.c_pathogen_id = toInteger(row.c_pathogen_id),
            r.c_reference_name = row.c_reference_name,
            r.c_reference_type = row.c_reference_type,
            r.c_authors = row.c_authors,
            r.c_title = row.c_title,
            r.c_year = row.c_year,
            r.c_volume = row.c_volume,
            r.c_issue = row.c_issue,
            r.c_pages = row.c_pages,
            r.c_journal_book_name = row.c_journal_book_name,
            r.c_publisher = row.c_publisher,
            r.c_publisher_location = row.c_publisher_location,
            r.c_book_editors = row.c_book_editors,
            r.c_isbn = row.c_isbn,
            r.c_university = row.c_university,
            r.c_university_location = row.c_university_location,
            r.c_degree = row.c_degree,
            r.c_pmid = row.c_pmid,
            r.c_url = row.c_url,
            r.c_file_name = row.c_file_name,
            r.c_gene_id = toInteger(row.c_gene_id),
            r.c_curation_flag = row.c_curation_flag,
            r.c_vaxjo_id = toInteger(row.c_vaxjo_id),
            r.c_dv_plasmid_id = row.c_dv_plasmid_id,
            r.c_vaxvec_id = toInteger(row.c_vaxvec_id),
            r.t_referencecol = row.t_referencecol,
            r.c_vaxanimodel_id = row.c_vaxanimodel_id,
            r.c_comment = row.c_comment;
    """,
    
    "Vaccine": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_vaccine.csv' AS row
        MERGE (v:Vaccine {{c_vaccine_id: toInteger(row.c_vaccine_id)}})
        SET v.c_vaccine_name = row.c_vaccine_name,
            v.c_type = row.c_type,
            v.c_is_combination_vaccine = row.c_is_combination_vaccine,
            v.c_description = row.c_description,
            v.c_adjuvant = row.c_adjuvant,
            v.c_storage = row.c_storage,
            v.c_pathogen_id = toInteger(row.c_pathogen_id),
            v.c_virulence = row.c_virulence,
            v.c_preparation = row.c_preparation,
            v.c_brand_name = row.c_brand_name,
            v.c_full_text = row.c_full_text,
            v.c_antigen = row.c_antigen,
            v.c_curation_flag = row.c_curation_flag,
            v.c_vector = row.c_vector,
            v.c_proper_name = row.c_proper_name,
            v.c_manufacturer = row.c_manufacturer,
            v.c_contraindication = row.c_contraindication,
            v.c_status = row.c_status,
            v.c_location_licensed = row.c_location_licensed,
            v.c_host_species = row.c_host_species,
            v.c_route = row.c_route,
            v.c_vo_id = row.c_vo_id,
            v.c_usage_age = row.c_usage_age,
            v.c_model_host = row.c_model_host,
            v.c_preservative = row.c_preservative,
            v.c_allergen = row.c_allergen,
            v.c_preparation_vo_id = row.c_preparation_vo_id,
            v.c_host_species2 = row.c_host_species2,
            v.c_cvx_code = row.c_cvx_code,
            v.c_cvx_desc = row.c_cvx_desc;
    """,
    
    "VaccineDetail": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_vaccine_detail.csv' AS row
        MERGE (vd:VaccineDetail {{c_vaccine_detail_id: toInteger(row.c_vaccine_detail_id)}})
        SET vd.c_vaccine_id = toInteger(row.c_vaccine_id),
            vd.c_pathogen_id = toInteger(row.c_pathogen_id),
            vd.c_type = row.c_type,
            vd.c_antigen = row.c_antigen,
            vd.c_model_host = row.c_model_host,
            vd.c_vector = row.c_vector,
            vd.c_allergen = row.c_allergen,
            vd.c_description = row.c_description,
            vd.c_preparation = row.c_preparation,
            vd.c_virulence = row.c_virulence,
            vd.c_full_text = row.c_full_text,
            vd.c_curation_flag = row.c_curation_flag,
            vd.c_contraindication = row.c_contraindication;
    """,
    
    "Vaxjo": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_vaxjo.csv' AS row
        MERGE (vj:Vaxjo {{c_vaxjo_id: toInteger(row.c_vaxjo_id)}})
        SET vj.c_components = row.c_components,
            vj.c_function = row.c_function,
            vj.c_dosage = row.c_dosage,
            vj.c_safety = row.c_safety,
            vj.c_vaxjo_vo_id = row.c_vaxjo_vo_id,
            vj.c_vaxjo_name = row.c_vaxjo_name,
            vj.c_submitted_by = row.c_submitted_by,
            vj.c_submission_date = row.c_submission_date,
            vj.c_curation_flag = row.c_curation_flag,
            vj.c_reviewer = row.c_reviewer,
            vj.c_preparation = row.c_preparation,
            vj.c_description = row.c_description,
            vj.c_alternative_names = row.c_alternative_names,
            vj.c_structure = row.c_structure,
            vj.c_appearance = row.c_appearance,
            vj.c_molecular_weight = row.c_molecular_weight,
            vj.c_storage = row.c_storage,
            vj.c_stage_dev = row.c_stage_dev,
            vj.c_location_licensed = row.c_location_licensed,
            vj.c_host_licensed = row.c_host_licensed,
            vj.c_host_licensed2 = row.c_host_licensed2;
    """,
    
    "Vaxvec": f"""
        LOAD CSV WITH HEADERS FROM '{CSV_PATH}t_vaxvec.csv' AS row
        MERGE (vv:Vaxvec {{c_vaxvec_id: toInteger(row.c_vaxvec_id)}})
        SET vv.c_vaxvec_name = row.c_vaxvec_name,
            vv.c_alternative_names = row.c_alternative_names,
            vv.c_advantage = row.c_advantage,
            vv.c_disadvantage = row.c_disadvantage,
            vv.c_safety = row.c_safety,
            vv.c_stability = row.c_stability,
            vv.c_host = row.c_host,
            vv.c_vaxvec_vo_id = row.c_vaxvec_vo_id,
            vv.c_description = row.c_description,
            vv.c_submitted_by = row.c_submitted_by,
            vv.c_submission_date = row.c_submission_date,
            vv.c_curation_flag = row.c_curation_flag,
            vv.c_preparation = row.c_preparation,
            vv.c_storage = row.c_storage,
            vv.c_function = row.c_function;
    """
}

RELATIONSHIP_QUERIES = {
    "VaccineDetailVaccine": f"""
        MATCH (vd:VaccineDetail), (v:Vaccine)
        WHERE vd.c_vaccine_id = v.c_vaccine_id
        MERGE (vd)-[:SPECIFIES]->(v);
    """,
    
    "ReferenceVaccine": f"""
        MATCH (r:Reference), (v:Vaccine)
        WHERE r.c_vaccine_id = v.c_vaccine_id OR 
              (r.c_pathogen_id = v.c_pathogen_id AND r.c_vaccine_id IS NOT NULL)
        MERGE (r)-[:MENTIONS]->(v);
    """,
    
    "VaccineAdjuvant": f"""
        MATCH (v:Vaccine), (a:Adjuvant)
        WHERE a.c_vaccine_id = v.c_vaccine_id
        MERGE (v)-[:CONTAINS_ADJUVANT]->(a);
    """,
    
    "VaccineHost": f"""
        MATCH (v:Vaccine), (h:Host)
        WHERE v.c_host_species = h.c_host_name OR 
              v.c_host_species2 = h.c_host_name OR 
              v.c_model_host = h.c_host_name
        MERGE (v)-[:HOSTED_IN]->(h);
    """,
    
    "VaccinePathogen": f"""
        MATCH (v:Vaccine), (p:Pathogen)
        WHERE v.c_pathogen_id = p.c_pathogen_id
        MERGE (v)-[:TARGETS]->(p);
    """,
    
    "AdjuvantVaxjo": f"""
        MATCH (a:Adjuvant), (vj:Vaxjo)
        WHERE a.c_adjuvant_vo_id = vj.c_vaxjo_vo_id
        MERGE (a)-[:REFERENCED_IN]->(vj);
    """,
    
    "AdjuvantVaxvec": f"""
        MATCH (a:Adjuvant), (vv:Vaxvec)
        WHERE a.c_adjuvant_vo_id = vv.c_vaxvec_vo_id
        MERGE (a)-[:REFERENCED_IN]->(vv);
    """,
    
    "AdjuvantPathogen": f"""
        MATCH (a:Adjuvant), (p:Pathogen)
        WHERE a.c_pathogen_id = p.c_pathogen_id
        MERGE (a)-[:TARGETS]->(p);
    """,
    
    "GeneEngineeringVaccine": f"""
        MATCH (ge:GeneEngineering), (v:Vaccine)
        WHERE ge.c_vaccine_id = v.c_vaccine_id
        MERGE (ge)-[:APPLIES_TO]->(v);
    """,
    
    "GeneGeneEngineering": f"""
        MATCH (g:Gene), (ge:GeneEngineering)
        WHERE g.c_gene_id = ge.c_gene_id
        MERGE (g)-[:MODIFIED_BY]->(ge);
    """,
    
    "GenePathogen": f"""
        MATCH (g:Gene), (p:Pathogen)
        WHERE g.c_pathogen_id = p.c_pathogen_id
        MERGE (g)-[:IS_FOUND_IN]->(p);
    """,
    
    "GeneHostGeneResponse": f"""
        MATCH (g:Gene), (hgr:HostGeneResponse)
        WHERE g.c_gene_id = hgr.c_gene_id
        MERGE (g)-[:ASSOCIATES_WITH_GENE]->(hgr);
    """,
    
    "HostGeneResponseHostResponse": f"""
        MATCH (hgr:HostGeneResponse), (hr:HostResponse)
        WHERE hgr.c_host_response_id = hr.c_host_response_id
        MERGE (hgr)-[:INCLUDES_GENE_RESPONSE]->(hr);
    """,
    
    "HostHostResponse": f"""
        MATCH (h:Host), (hr:HostResponse)
        WHERE h.c_host_id = hr.c_host_id
        MERGE (h)-[:GENERATES_RESPONSE]->(hr);
    """,
    
    "HostResponseVaccine": f"""
        MATCH (hr:HostResponse), (v:Vaccine)
        WHERE hr.c_vaccine_id = v.c_vaccine_id
        MERGE (hr)-[:RESPONSE_TO]->(v);
    """,
    
    "ReferencePathogen": f"""
        MATCH (r:Reference), (p:Pathogen)
        WHERE r.c_pathogen_id = p.c_pathogen_id
        MERGE (r)-[:DESCRIBES]->(p);
    """,
    
    "ReferenceGene": f"""
        MATCH (r:Reference), (g:Gene)
        WHERE r.c_gene_id = g.c_gene_id
        MERGE (r)-[:MENTIONS]->(g);
    """,
    
    "ReferenceVaxjo": f"""
        MATCH (r:Reference), (vj:Vaxjo)
        WHERE r.c_vaxjo_id = vj.c_vaxjo_id
        MERGE (r)-[:CITES]->(vj);
    """,
    
    "ReferenceVaxvec": f"""
        MATCH (r:Reference), (vv:Vaxvec)
        WHERE r.c_vaxvec_id = vv.c_vaxvec_id
        MERGE (r)-[:REFERENCES]->(vv);
    """
}

def execute_queries(driver, queries):
    """
    Execute a collection of Cypher queries in a single Neo4j session.

    This helper function runs multiple Cypher queries sequentially, providing 
    progress tracking and error-tolerant execution. It's useful for batch 
    operations like creating constraints, importing data, or establishing 
    relationships.

    Args:
        driver (GraphDatabase.driver): Active Neo4j database driver
        queries (dict): Dictionary of query names and their corresponding Cypher queries

    Raises:
        Exception: If a query fails during execution
    """    """Helper function to execute a dictionary of Cypher queries."""
    with driver.session() as session:
        for key, query in queries.items():
            print(f"Executing: {key}")
            session.run(query)

def import_ontology_complete(driver):
    """
    Import a comprehensive ontology into the Neo4j graph database using n10s.

    This function handles the complete ontology import process, including:
    - Creating necessary constraints
    - Configuring n10s graph configuration
    - Importing RDF/XML ontology from a remote URL
    - Handling potential import errors
    - Providing detailed import statistics

    Args:
        driver (GraphDatabase.driver): Active Neo4j database driver

    Notes:
        - Uses the Vaccine Ontology (VO) from GitHub
        - Configures n10s with specific handling strategies for RDF import
        - Prints detailed import progress and results
    """
    with driver.session() as session:
        try:
            # Create the required constraint if it doesn't exist
            print("Creating required constraint for n10s...")
            try:
                session.run("CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE")
                print("Constraint created")
            except Exception as constraint_error:
                # If constraint already exists, this will fail, which is fine
                print(f"Note: {str(constraint_error)}")
                print("Constraint may already exist, continuing...")
            
            # Reset any previous graph config
            session.run("MATCH (n:_GraphConfig) DELETE n")
            print("Cleared previous graph config")
            
            # Configure n10s properly
            session.run("""
                CALL n10s.graphconfig.init({
                  handleVocabUris: 'MAP',
                  handleMultipleRelsPerType: 'OVERWRITE',
                  handleRDFTypes: 'NODES'
                })
            """)
            print("n10s configured successfully")
            
            # Import from the GitHub URL
            owl_url = "https://raw.githubusercontent.com/vaccineontology/VO/master/vo.owl"
            
            print(f"Importing ontology from URL: {owl_url}")
            import_result = session.run(
                "CALL n10s.rdf.import.fetch($url, 'RDF/XML') YIELD triplesParsed RETURN triplesParsed",
                {"url": owl_url}
            )
            
            for record in import_result:
                print(f"Import result: {record['triplesParsed']} triples parsed")
            
            # Check what was imported
            count_result = session.run("MATCH (n:Resource) RETURN count(n) as resourceCount")
            for record in count_result:
                print(f"Imported {record['resourceCount']} Resource nodes")
            
        except Exception as e:
            print(f"Error during ontology import: {str(e)}")
            import traceback
            print(traceback.format_exc())

# Function to execute queries
def import_data():
    """
    Orchestrate the complete data import process for the Neo4j graph database.

    This function manages the entire import workflow:
    1. Establish database connection
    2. Create node constraints
    3. Import CSV data into nodes
    4. Establish relationships between nodes

    Raises:
        Exception: If any stage of the import process fails
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    # Step 1: Create Constraints
    execute_queries(driver, CONSTRAINT_QUERIES)
    print("Constraints created.")
    
    # Step 2: Import CSV Data into Nodes
    execute_queries(driver, CREATE_QUERIES)
    print("CSV data imported into nodes.")

    # Step 3: Establish Relationships
    execute_queries(driver, RELATIONSHIP_QUERIES)
    print("Relationships created.")

    driver.close()

# Run the import function
if __name__ == "__main__":
    import_data()