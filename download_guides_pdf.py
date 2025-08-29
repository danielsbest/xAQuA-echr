import os
import time
import requests

# Set the country code and backup
country_code = "ita"  # Primary country code
backup_country_code = ""  # Backup country code (can be empty "" for most countries, fra for fre, rum for ron...)

# Create output directory relative to project root
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "parse_control", "marker-pdf", country_code)
os.makedirs(output_dir, exist_ok=True)

# Hardcoded list of English guide IDs   
guide_ids_eng = [
    "guide_art_1_eng",
    "guide_art_2_eng",
    "guide_art_3_eng",
    "guide_art_4_eng",
    "guide_art_5_eng",
    "guide_art_6_civil_eng",
    "guide_art_6_criminal_eng",
    "guide_art_7_eng",
    "guide_art_8_eng",
    "guide_art_9_eng",
    "guide_art_10_eng",
    "guide_art_11_eng",
    "guide_art_12_eng",
    "guide_art_13_eng",
    "guide_art_14_art_1_protocol_12_eng",
    "guide_art_15_eng",
    "guide_art_17_eng",
    "guide_art_18_eng",
    "admissibility_guide_eng", # also called guide_art_34_35_eng
    "guide_art_46_eng",
    "guide_art_1_protocol_1_eng",
    "guide_art_2_protocol_1_eng",
    "guide_art_3_protocol_1_eng",
    "guide_art_2_protocol_4_eng",
    "guide_art_3_protocol_4_eng",
    "guide_art_4_protocol_4_eng",
    "guide_art_1_protocol_7_eng",
    "guide_art_2_protocol_7_eng",
    "guide_art_4_protocol_7_eng",
    "guide_data_protection_eng",
    "guide_environment_eng",
    "guide_immigration_eng",
    "guide_mass_protests_eng",
    "guide_prisoners_rights_eng",
    "guide_lgbti_rights_eng",
    "guide_social_rights_eng",
    "guide_terrorism_eng",
    "guide_rights_of_the_child_eng",
    "guide_eu_law_in_echr_case-law_eng",
]

def check_guide_availability(guide_id, base_url):
    """Check if a guide is available by testing the URL"""
    url = base_url.format(guide_id)
    try:
        response = requests.head(url, timeout=10)
        # Check if it's a PDF and not a 404 redirect
        return (response.status_code == 200 and 
                "page-404" not in response.url and
                "application/pdf" in response.headers.get("content-type", "").lower())
    except:
        return False

def download_pdf_direct(guide_id, base_url, country_code):
    """Download PDF directly from the guide URL"""
    url = base_url.format(guide_id)
    
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, timeout=30)
        
        # Check if we got redirected to 404
        if "page-404" in response.url:
            print(f"✗ Guide redirected to 404: {guide_id}")
            return "unavailable"
            
        if response.status_code == 200:
            # Check if content is actually a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or response.content.startswith(b'%PDF'):
                # Always use primary country code in filename
                filename = f"{guide_id.replace(guide_id.split('_')[-1], country_code)}.pdf"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Downloaded: {filename} ({len(response.content)} bytes)")
                return "success"
            else:
                print(f"✗ Content is not a PDF for {guide_id} (content-type: {content_type})")
                return "error"
        else:
            print(f"✗ Failed to download PDF for {guide_id}: HTTP {response.status_code}")
            if response.status_code == 404:
                return "unavailable"
            else:
                return "error"
            
    except Exception as e:
        print(f"✗ Error downloading {guide_id}: {str(e)}")
        return "error"

def main():
    # Base URL format
    base_url = "https://ks.echr.coe.int/documents/d/echr-ks/{}"
    
    print(f"Downloading PDF guides for country code: {country_code.upper()}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    downloaded_count = 0
    unavailable_count = 0
    error_count = 0
    backup_used_count = 0
    unavailable_guides = []  # Track which guides were unavailable
    downloaded_guides = []  # Track which guides were successfully downloaded
    
    for guide_id in guide_ids_eng:
        # Convert to localized guide ID
        localized_guide_id = guide_id.replace("_eng", f"_{country_code}").replace("_ENG", f"_{country_code}")
        
        print(f"Processing: {localized_guide_id}")
        
        # Try to download the PDF directly
        result = download_pdf_direct(localized_guide_id, base_url, country_code)
        
        if result == "unavailable" and backup_country_code:
            # Try backup country code
            backup_guide_id = guide_id.replace("_eng", f"_{backup_country_code}").replace("_ENG", f"_{backup_country_code}")
            print(f"Trying backup: {backup_guide_id}")
            result = download_pdf_direct(backup_guide_id, base_url, backup_country_code)
            if result == "success":
                backup_used_count += 1
        
        if result == "success":
            downloaded_count += 1
            downloaded_guides.append(localized_guide_id)
        elif result == "unavailable":
            unavailable_count += 1
            # Store the original English guide ID for clarity
            unavailable_guides.append(localized_guide_id)
        else:  # error
            error_count += 1
        
        time.sleep(0.5)  # Be respectful to the server
    
    print("\n" + "=" * 60)
    print(f"Download Summary for {country_code.upper()} (backup: {backup_country_code.upper() if backup_country_code else 'None'}):")
    print(f"- Total guides processed: {len(guide_ids_eng)}")
    print(f"- Successfully downloaded: {downloaded_count}")
    print(f"- Downloaded using backup code: {backup_used_count}")
    print(f"- Unavailable (404/not found): {unavailable_count}")
    print(f"- Failed with errors: {error_count}")
    print(f"- PDFs saved to: {output_dir}")
    
    # Display unavailable guides
    if unavailable_guides:
        print(f"\nUnavailable guides ({unavailable_count}):")
        for guide in unavailable_guides:
            print(f"  - {guide}")
        print(f"\nNote: admissibility guide often has naming: 'Admissibility_guide_{backup_country_code}.pdf' -> download manually")
    
    # Display successfully downloaded guides
    if downloaded_guides:
        print(f"\nSuccessfully downloaded guides ({downloaded_count}):")
        for guide in downloaded_guides:
            print(f"  - {guide}")

if __name__ == "__main__":
    main()