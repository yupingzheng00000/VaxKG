from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
# environment: neo4j-env\Scripts\activate  

# CSV file path prefix
CSV_PATH = "file:///"

RELATIONSHIP_QUERIES_VO_MAPPED = {
    "VaccineDetailVaccine": f"""
        MATCH (vd:VaccineDetail)-[:SPECIFIES]->(v:Vaccine)
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // Assuming VO_VaccineProduct for Vaccine
        MATCH (original_vd:VaccineDetail {{c_vaccine_detail_id: vd.c_vaccine_detail_id}}) // Match original VaccineDetail node
        MERGE (original_vd)-[:SPECIFIES_VO]->(vo_v); // New VO-aligned relationship
    """,

    "ReferenceVaccine": f"""
        MATCH (r:Reference)-[:MENTIONS]->(v:Vaccine)
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // Assuming VO_VaccineProduct for Vaccine
        MATCH (original_r:Reference {{c_reference_id: r.c_reference_id}}) // Match original Reference node
        MERGE (original_r)-[:MENTIONS_VO]->(vo_v); // New VO-aligned relationship
    """,

    "VaccineAdjuvant": f"""
        MATCH (v:Vaccine)-[:CONTAINS_ADJUVANT]->(a:Adjuvant)
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // VO Vaccine
        MATCH (vo_a:VO_VaccineAdjuvant {{uri: a.c_adjuvant_id}}) // Assuming VO_VaccineAdjuvant for Adjuvant
        MERGE (vo_v)-[:CONTAINS_ADJUVANT_VO]->(vo_a); // VO-aligned relationship - using descriptive name
    """,

    "VaccineHost": f"""
        MATCH (v:Vaccine)-[:HOSTED_IN]->(h:Host)
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // VO Vaccine
        MATCH (vo_h:VO_Host {{uri: h.c_host_id}}) // Assuming VO_Host for Host
        MERGE (vo_v)-[:HOSTED_IN_VO]->(vo_h); // VO-aligned relationship - using descriptive name
    """,

    "VaccinePathogen": f"""
        MATCH (v:Vaccine)-[:TARGETS]->(p:Pathogen)
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // VO Vaccine
        MATCH (vo_p:VO_Pathogen {{uri: p.c_pathogen_id}}) // Assuming VO_Pathogen for Pathogen
        MERGE (vo_v)-[:TARGETS_VO]->(vo_p); // VO-aligned relationship - using descriptive name
    """,

    "AdjuvantVaxjo": f"""
        MATCH (a:Adjuvant)-[:REFERENCED_IN]->(vj:Vaxjo)
        MATCH (vo_a:VO_VaccineAdjuvant {{uri: a.c_adjuvant_id}}) // VO Adjuvant
        MATCH (original_vj:Vaxjo {{c_vaxjo_id: vj.c_vaxjo_id}}) // Match original Vaxjo node
        MERGE (vo_a)-[:REFERENCED_IN_VO]->(original_vj); // VO-aligned relationship to original Vaxjo
    """,

    "AdjuvantVaxvec": f"""
        MATCH (a:Adjuvant)-[:REFERENCED_IN]->(vv:Vaxvec)
        MATCH (vo_a:VO_VaccineAdjuvant {{uri: a.c_adjuvant_id}}) // VO Adjuvant
        MATCH (original_vv:Vaxvec {{c_vaxvec_id: vv.c_vaxvec_id}}) // Match original Vaxvec node
        MERGE (vo_a)-[:REFERENCED_IN_VO]->(original_vv); // VO-aligned relationship to original Vaxvec
    """,

    "AdjuvantPathogen": f"""
        MATCH (a:Adjuvant)-[:TARGETS]->(p:Pathogen)
        MATCH (vo_a:VO_VaccineAdjuvant {{uri: a.c_adjuvant_id}}) // VO Adjuvant
        MATCH (vo_p:VO_Pathogen {{uri: p.c_pathogen_id}}) // VO Pathogen
        MERGE (vo_a)-[:TARGETS_VO]->(vo_p); // VO-aligned relationship - using descriptive name
    """,

    "GeneEngineeringVaccine": f"""
        MATCH (ge:GeneEngineering)-[:APPLIES_TO]->(v:Vaccine)
        MATCH (vo_ge:VO_GeneEngineering {{uri: ge.c_gene_engineering_id}}) // Assuming VO_GeneEngineering for GeneEngineering
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // VO Vaccine
        MERGE (vo_ge)-[:APPLIES_TO_VO]->(vo_v); // VO-aligned relationship - using descriptive name
    """,

    "GeneGeneEngineering": f"""
        MATCH (g:Gene)-[:MODIFIED_BY]->(ge:GeneEngineering)
        MATCH (vo_g:VO_Gene {{uri: g.c_gene_id}}) // VO Gene
        MATCH (vo_ge:VO_GeneEngineering {{uri: ge.c_gene_engineering_id}}) // VO GeneEngineering
        MERGE (vo_g)-[:MODIFIED_BY_VO]->(vo_ge); // VO-aligned relationship - using descriptive name
    """,

    "GenePathogen": f"""
        MATCH (g:Gene)-[:IS_FOUND_IN]->(p:Pathogen)
        MATCH (vo_g:VO_Gene {{uri: g.c_gene_id}}) // VO Gene
        MATCH (vo_p:VO_Pathogen {{uri: p.c_pathogen_id}}) // VO Pathogen
        MERGE (vo_g)-[:IS_FOUND_IN_VO]->(vo_p); // VO-aligned relationship - using descriptive name
    """,

    "GeneHostGeneResponse": f"""
        MATCH (g:Gene)-[:ASSOCIATES_WITH_GENE]->(hgr:HostGeneResponse)
        MATCH (vo_g:VO_Gene {{uri: g.c_gene_id}}) // VO Gene
        MATCH (original_hgr:HostGeneResponse {{c_host_gene_response_id: hgr.c_host_gene_response_id}}) // Match original HostGeneResponse node
        MERGE (vo_g)-[:ASSOCIATES_WITH_GENE_VO]->(original_hgr); // VO-aligned relationship to original HostGeneResponse
    """,

    "HostGeneResponseHostResponse": f"""
        MATCH (hgr:HostGeneResponse)-[:INCLUDES_GENE_RESPONSE]->(hr:HostResponse)
        MATCH (original_hgr:HostGeneResponse {{c_host_gene_response_id: hgr.c_host_gene_response_id}}) // Match original HostGeneResponse node
        MATCH (vo_hr:VO_HostResponse {{uri: hr.c_host_response_id}}) // Assuming VO_HostResponse for HostResponse
        MERGE (original_hgr)-[:INCLUDES_GENE_RESPONSE_VO]->(vo_hr); // VO-aligned relationship to VO HostResponse
    """,

    "HostHostResponse": f"""
        MATCH (h:Host)-[:GENERATES_RESPONSE]->(hr:HostResponse)
        MATCH (vo_h:VO_Host {{uri: h.c_host_id}}) // VO Host
        MATCH (vo_hr:VO_HostResponse {{uri: hr.c_host_id}}) // Assuming VO_HostResponse for HostResponse (typo in original query - should be hr.c_host_response_id?)
        MERGE (vo_h)-[:GENERATES_RESPONSE_VO]->(vo_hr); // VO-aligned relationship - using descriptive name
    """,

    "HostResponseVaccine": f"""
        MATCH (hr:HostResponse)-[:RESPONSE_TO]->(v:Vaccine)
        MATCH (vo_hr:VO_HostResponse {{uri: hr.c_host_response_id}}) // VO HostResponse
        MATCH (vo_v:VO_VaccineProduct {{uri: v.c_vaccine_id}}) // VO Vaccine
        MERGE (vo_hr)-[:RESPONSE_TO_VO]->(vo_v); // VO-aligned relationship - using descriptive name
    """,

    "ReferencePathogen": f"""
        MATCH (r:Reference)-[:DESCRIBES]->(p:Pathogen)
        MATCH (original_r:Reference {{c_reference_id: r.c_reference_id}}) // Match original Reference node
        MATCH (vo_p:VO_Pathogen {{uri: p.c_pathogen_id}}) // VO Pathogen
        MERGE (original_r)-[:DESCRIBES_VO]->(vo_p); // VO-aligned relationship to VO Pathogen
    """,

    "ReferenceGene": f"""
        MATCH (r:Reference)-[:MENTIONS]->(g:Gene)
        MATCH (original_r:Reference {{c_reference_id: r.c_reference_id}}) // Match original Reference node
        MATCH (vo_g:VO_Gene {{uri: g.c_gene_id}}) // VO Gene
        MERGE (original_r)-[:MENTIONS_VO]->(vo_g); // VO-aligned relationship to VO Gene
    """
}

def import_ontology_complete(driver):
    """Import ontology with proper setup, error handling, and mapping"""
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
    """Import ontology with proper setup, error handling, and mapping"""
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
            map_relationships_vo_aligned(session, RELATIONSHIP_QUERIES_VO_MAPPED)

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
    """Updates Resource nodes to use human-readable property keys instead of IAO codes."""
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
    """Maps VO Resource properties to Vaccine node with manually remapped keys and creates a relationship to the VO Resource node."""
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
    """Maps VO and UBPROP Resource properties to Pathogen nodes with manually remapped keys (prefixed with 'vo_')
    and creates a relationship to the corresponding VO Resource node, based on c_taxon_id matching."""
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
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:        
        import_ontology_complete(driver)
        map_ontology(driver)
        with driver.session() as session:
            # Update Resource node properties to human-readable labels
            update_resource_properties(session)
            # # Map Vaccine nodes as before
            # map_vaccine_nodes(session)
            # map_pathogen_nodes(session)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

    driver.close()

# Run the import function
if __name__ == "__main__":
    import_data()