from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
URI = os.getenv("URI")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
# environment: neo4j-env\Scripts\activate  

# CSV file path prefix
CSV_PATH = "file:///"

def import_ontology_complete(driver):
    """
    Import the Vaccine Ontology (VO) from a remote GitHub URL into a Neo4j graph database.

    This function performs the following key steps:
    1. Creates a unique constraint for Resource nodes
    2. Resets previous graph configuration
    3. Configures n10s (Neosemantics) for RDF import
    4. Fetches and imports the OWL ontology from a GitHub URL
    5. Handles potential errors during the import process

    Args:
        driver (GraphDatabase.driver): An active Neo4j database driver connection

    Raises:
        Exception: If there are issues with constraint creation, graph configuration, or ontology import
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

def map_ontology(driver):
    """
    Map imported ontology nodes to domain-specific nodes with proper ontological relationships.

    This function performs domain-specific node mapping:
    1. Verifies the number of imported Resource nodes
    2. Maps Vaccine, Pathogen, and potentially other domain-specific nodes
    3. Provides detailed logging of mapping processes

    Args:
        driver (GraphDatabase.driver): An active Neo4j database driver connection

    Raises:
        Exception: If there are issues during the ontology mapping process
    """
    with driver.session() as session:
        try:
            # Check what was imported
            count_result = session.run("MATCH (n:Resource) RETURN count(n) as resourceCount")
            for record in count_result:
                print(f"Imported {record['resourceCount']} Resource nodes")

            # Execute node mapping queries
            print("\nMapping domain nodes to ontology...")

            map_vaccine_nodes(session) # Map Vaccine nodes
            # map_gene_nodes(session)     # Map Gene nodes
            map_pathogen_nodes(session) # Map Pathogen nodes
            # map_adjuvant_nodes(session) # Map Adjuvant nodes (add this new function)
            # map_host_nodes(session)     # Map Host nodes (add this new function)
            # map_host_response_nodes(session) # Map HostResponse nodes (add this new function)
            # map_gene_engineering_nodes(session) # Map GeneEngineering nodes (add this new function)


            print("\nNode mapping completed successfully, starting relationship mapping...")

            # Execute relationship mapping queries - VO-aligned relationships
            # map_relationships_vo_aligned(session, RELATIONSHIP_QUERIES_VO_MAPPED)

            print("\nMapping completed successfully")

        except Exception as e:
            print(f"Error during ontology import: {str(e)}")
            import traceback
            print(traceback.format_exc())

def map_vaccine_nodes_gemini_way(session):
    """Maps Vaccine nodes using c_vo_id and also to VO_0000001 Resource."""
    print("Mapping Vaccine nodes using c_vo_id and also to VO_0000001 Resource...")
    try:
        result = session.run("""
            MATCH (v:Vaccine)
            WHERE NOT isEmpty(v.c_vo_id)
            MATCH (vaccine_class_resource:Resource)
            WHERE last(split(vaccine_class_resource.uri, '/')) = v.c_vo_id
            MATCH (general_vaccine_resource:Resource) // Match Resource node for VO_0000001
            WHERE general_vaccine_resource.uri = 'http://purl.obolibrary.org/obo/VO_0000001'
            MERGE (vo_v:VO_VaccineProduct {uri: v.c_vaccine_id})
            SET vo_v.name = v.c_vaccine_name,
                vo_v.description = v.c_description
            MERGE (vo_v)-[:INSTANCE_OF]->(vaccine_class_resource) // Instance of specific vaccine class
            MERGE (vo_v)-[:INSTANCE_OF]->(general_vaccine_resource)   // Instance of general Vaccine class (VO_0000001)
            MERGE (v)-[:VO_REPRESENTATION]->(vo_v)
            RETURN count(*) as mappedCount
        """)
        for record in result:
            print(f"Mapped {record['mappedCount']} Vaccine nodes to specific and VO_0000001 Resources")
        unmapped_count_result = session.run("""
            MATCH (v:Vaccine)
            WHERE isEmpty(v.c_vo_id)
            RETURN count(*) as unmappedCount
        """)
        for record in unmapped_count_result:
            print(f"Warning: {record['unmappedCount']} Vaccine nodes were not mapped (missing c_vo_id).")

    except Exception as e:
        print(f"Error mapping Vaccine nodes to specific and VO_0000001 Resources: {str(e)}")

def update_resource_properties(session):
    """
    Update Resource nodes by transforming complex IAO and UBPROP codes to human-readable properties.

    This function performs two key operations:
    1. Creates new, more readable property keys from existing IAO and UBPROP coded properties
    2. Optionally removes the original coded properties to clean up the graph

    Key transformations include:
    - IAO_0000111 → editor_preferred_label
    - IAO_0000115 → definition
    - VO_0003099 → trade_name
    And many more similar transformations

    Args:
        session (neo4j.Session): An active Neo4j database session

    Raises:
        Exception: If there are issues during property transformation
    """
    print("Updating Resource node properties to human-readable labels...")
    try:
        # Copy values from IAO properties to new human-readable keys
        session.run("""
            MATCH (r:Resource)
            SET r.editor_preferred_label = r.IAO_0000111,
                r.example_of_usage = r.IAO_0000112,
                r.has_curation_status = r.IAO_0000114,
                r.definition = r.IAO_0000115,
                r.editor_note = r.IAO_0000116,
                r.term_editor = r.IAO_0000117,
                r.alternative_term = r.IAO_0000118,
                r.definition_source = r.IAO_0000119,
                r.curator_note = r.IAO_0000232,
                r.term_tracker_item = r.IAO_0000233,
                r.imported_from = r.IAO_0000412,
                r.violin_vaccine_id = r.VO_0001818,
                r.trade_name = r.VO_0003099,
                r.fda_vaccine_indications = r.VO_0003160,
                r.vaccine_package_insert_pdf_url = r.VO_0003161,
                r.vaccine_stn = r.VO_0003162,
                r.taxon_notes = r.UBPROP_0000008,
                r.external_definition = r.UBPROP_0000001,
                r.axiom_lost_from_external_ontology = r.UBPROP_0000002,
                r.homology_notes = r.UBPROP_0000003
        """)
        # Optionally remove the original IAO properties if they are no longer needed
        session.run("""
            MATCH (r:Resource)
            REMOVE r.IAO_0000111, r.IAO_0000112, r.IAO_0000114, r.IAO_0000115,
                   r.IAO_0000116, r.IAO_0000117, r.IAO_0000118, r.IAO_0000119,
                   r.IAO_0000232, r.IAO_0000233, r.IAO_0000412,
                   r.VO_0001818, r.VO_0003099, r.VO_0003160, r.VO_0003161, r.VO_0003162,
                   r.UBPROP_0000008, r.UBPROP_0000001, r.UBPROP_0000002, r.UBPROP_0000003
        """)
        print("Resource node properties updated successfully.")
    except Exception as e:
        print(f"Error updating Resource node properties: {str(e)}")

def map_vaccine_nodes(session):
    """
    Map Vaccine nodes to their corresponding Vaccine Ontology (VO) Resource representations.

    This function:
    1. Finds Vaccine nodes with a valid c_vo_id
    2. Locates corresponding Resource nodes in the ontology
    3. Remaps VO-specific properties to more readable keys
    4. Creates VO_REPRESENTATION relationships
    5. Logs mapping details and potential unmapped nodes

    Args:
        session (neo4j.Session): An active Neo4j database session

    Raises:
        Exception: If there are issues during vaccine node mapping
    """
    print("Mapping Vaccine nodes by remapping VO Resource properties and creating relationships...")
    try:
        result = session.run("""
            MATCH (v:Vaccine)
            WHERE v.c_vo_id IS NOT NULL AND v.c_vo_id <> ''
            MATCH (resource:Resource)
            WHERE last(split(resource.uri, '/')) = v.c_vo_id
            WITH v, resource
            SET v.vo_representation_uri = resource.uri,
                v.vo_editor_preferred_label = resource.IAO_0000111,
                v.vo_example_of_usage = resource.IAO_0000112,
                v.vo_has_curation_status = resource.IAO_0000114,
                v.vo_definition = resource.IAO_0000115,
                v.vo_editor_note = resource.IAO_0000116,
                v.vo_term_editor = resource.IAO_0000117,
                v.vo_alternative_term = resource.IAO_0000118,
                v.vo_definition_source = resource.IAO_0000119,
                v.vo_curator_note = resource.IAO_0000232,
                v.vo_term_tracker_item = resource.IAO_0000233,
                v.vo_imported_from = resource.IAO_0000412,
                v.vo_violin_vaccine_id = resource.VO_0001818,
                v.vo_trade_name = resource.VO_0003099,
                v.vo_fda_vaccine_indications = resource.VO_0003160,
                v.vo_vaccine_package_insert_pdf_url = resource.VO_0003161,
                v.vo_vaccine_stn = resource.VO_0003162,
                v.vo_taxon_notes = resource.UBPROP_0000008,
                v.vo_external_definition = resource.UBPROP_0000001,
                v.vo_axiom_lost_from_external_ontology = resource.UBPROP_0000002,
                v.vo_homology_notes = resource.UBPROP_0000003
            MERGE (v)-[:VO_REPRESENTATION]->(resource)
            RETURN count(*) as mappedCount
        """)
        for record in result:
            print(f"Mapped {record['mappedCount']} Vaccine nodes with remapped VO properties and created relationships")
        
        # Debug: Count the VO_REPRESENTATION relationships created
        rel_count_result = session.run("""
            MATCH (:Vaccine)-[r:VO_REPRESENTATION]->(:Resource)
            RETURN count(r) as relCount
        """)
        for record in rel_count_result:
            print(f"Total VO_REPRESENTATION relationships in the graph: {record['relCount']}")
        
        unmapped_count_result = session.run("""
            MATCH (v:Vaccine)
            WHERE v.c_vo_id IS NULL OR v.c_vo_id = ''
            RETURN count(*) as unmappedCount
        """)
        for record in unmapped_count_result:
            print(f"Warning: {record['unmappedCount']} Vaccine nodes were not mapped (missing c_vo_id).")
    
    except Exception as e:
        print(f"Error mapping Vaccine nodes with remapped VO properties and relationship creation: {str(e)}")


def map_pathogen_nodes(session):
    """
    Map Pathogen nodes to their corresponding Taxonomy Resource representations.

    This function performs the following key operations:
    1. Identifies Pathogen nodes with a valid c_taxon_id
    2. Locates corresponding Taxonomy Resource nodes
    3. Remaps ontology-specific properties to more readable keys
    4. Creates VO_REPRESENTATION relationships between Pathogen and Resource nodes

    Mapping uses the following key logic:
    - Match Pathogen nodes by c_taxon_id
    - Find corresponding Resource nodes using NCBITaxon_ prefix
    - Transfer important ontological properties
    - Create a representation relationship

    Args:
        session (neo4j.Session): An active Neo4j database session

    Raises:
        Exception: If there are issues during pathogen node mapping
    """
    print("Mapping Pathogen nodes by remapping VO and UBPROP Resource properties and creating relationships...")
    try:
        result = session.run("""
            MATCH (p:Pathogen)
            WHERE p.c_taxon_id IS NOT NULL AND p.c_taxon_id <> ''
            MATCH (resource:Resource)
            WHERE last(split(resource.uri, '/')) = 'NCBITaxon_' + p.c_taxon_id
            WITH p, resource
            SET p.vo_representation_uri = resource.uri,
                p.vo_editor_preferred_label = resource.IAO_0000111,
                p.vo_example_of_usage = resource.IAO_0000112,
                p.vo_has_curation_status = resource.IAO_0000114,
                p.vo_definition = resource.IAO_0000115,
                p.vo_editor_note = resource.IAO_0000116,
                p.vo_term_editor = resource.IAO_0000117,
                p.vo_alternative_term = resource.IAO_0000118,
                p.vo_definition_source = resource.IAO_0000119,
                p.vo_curator_note = resource.IAO_0000232,
                p.vo_term_tracker_item = resource.IAO_0000233,
                p.vo_imported_from = resource.IAO_0000412,
                p.vo_violin_vaccine_id = resource.VO_0001818,
                p.vo_trade_name = resource.VO_0003099,
                p.vo_fda_vaccine_indications = resource.VO_0003160,
                p.vo_vaccine_package_insert_pdf_url = resource.VO_0003161,
                p.vo_vaccine_stn = resource.VO_0003162,
                p.vo_taxon_notes = resource.UBPROP_0000008,
                p.vo_external_definition = resource.UBPROP_0000001,
                p.vo_axiom_lost_from_external_ontology = resource.UBPROP_0000002,
                p.vo_homology_notes = resource.UBPROP_0000003
            MERGE (p)-[:VO_REPRESENTATION]->(resource)
            RETURN count(*) as mappedCount
        """)
        for record in result:
            print(f"Mapped {record['mappedCount']} Pathogen nodes and created corresponding VO_REPRESENTATION relationships")
        
        # Debug: Count the VO_REPRESENTATION relationships created for Pathogen nodes
        rel_count_result = session.run("""
            MATCH (:Pathogen)-[r:VO_REPRESENTATION]->(:Resource)
            RETURN count(r) as relCount
        """)
        for record in rel_count_result:
            print(f"Total VO_REPRESENTATION relationships for Pathogen in the graph: {record['relCount']}")
    
    except Exception as e:
        print(f"Error mapping Pathogen nodes with remapped VO and UBPROP properties: {str(e)}")

def map_relationships_vo_aligned(session, relationship_queries):
    """Executes relationship mapping queries that are aligned with VO concepts."""
    print("\nMapping relationships using VO alignment...")
    for name, query in relationship_queries.items():
        print(f"Executing relationship query: {name}")
        try:
            result = session.run(query)
            for record in result:
                mapped_count = record.get("mappedCount", None)  # Handle cases where count is not returned
                if mapped_count is not None:
                    print(f"  Mapped {mapped_count} relationships for {name}")
                else:
                    print(f"  Query for {name} executed successfully (count not available in query)")
        except Exception as e:
            print(f"  Error executing relationship query {name}: {str(e)}")

def import_data():
    """
    Main orchestration function to import and map ontology data.

    This function:
    1. Establishes a Neo4j database connection
    2. Imports the complete ontology
    3. Maps domain-specific nodes
    4. Updates Resource node properties
    5. Handles any exceptions during the process

    Raises:
        Exception: If there are issues during the entire import process
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:        
        import_ontology_complete(driver)
        map_ontology(driver)
        with driver.session() as session:
            # Update Resource node properties to human-readable labels
            update_resource_properties(session)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

    driver.close()

# Run the import function
if __name__ == "__main__":
    import_data()