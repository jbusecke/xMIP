import warnings
import numpy as np
import xarray as xr


def _default_merge_dict():
    return {
        "North Atlantic Ocean": [
            "Caribbean Sea",
            "Gulf of Mexico",
            "Labrador Sea",
            "Hudson Bay",
            "Baffin Bay",
            "Norwegian Sea",
            "Greenland Sea",
            "Bay of Biscay",
            "Norwegian Sea",
            "Greenland Sea",
            "Gulf of Guinea",
            "Irish Sea",
            "North Sea",
            "Bahía de Campeche",
            "Davis Strait",
            "Sargasso Sea",
            "Hudson Strait",
            "English Channel",
            "Gulf of Honduras",
            "Bristol Channel",
            "Inner Seas",
            "Straits of Florida",
            "Gulf of Saint Lawrence",
            "Bay of Fundy",
            "Melville Bay",
            "Gulf of Maine",
            "Chesapeake Bay",
            "Amazon River",
            "James Bay",
            "Ungava Bay",
        ],
        "South Atlantic Ocean": ["Río de la Plata", "Golfo San Jorge"],
        "North Pacific Ocean": [
            "Philippine Sea",
            "Gulf of Alaska",
            "Sea of Okhotsk",
            "East China Sea",
            "Yellow Sea",
            "Bering Sea",
            "Golfo de California",
            "Korea Strait",
            "Cook Inlet",
            "Bristol Bay",
            "Shelikhova Gulf",
            "Bo Hai",
            "Golfo de Panamá",
            "Yangtze River",
            "Columbia River",
            "Sea of Japan",
            "Inner Sea",
        ],
        "South Pacific Ocean": [
            "Coral Sea",
            "Tasman Sea",
            "Bay of Plenty",
            "Bismarck Sea",
            "Solomon Sea",
            "Great Barrier Reef",
        ],
        "Maritime Continent": [
            "Celebes Sea",
            "Sulu Sea",
            "Banda Sea",
            "Luzon Strait",
            "Java Sea",
            "Arafura Sea",
            "Timor Sea",
            "Gulf of Thailand",
            "Gulf of Carpentaria",
            "Molucca Sea",
            "Gulf of Tonkin",
            "Strait of Malacca",
            "Strait of Singapore",
            "Makassar Strait",
            "Ceram Sea",
            "Taiwan Strait",
            "South China Sea",
        ],
        "INDIAN OCEAN": [
            "Mozambique Channel",
            "Bay of Bengal",
            "Arabian Sea",
            "Persian Gulf",
            "Andaman Sea",
            "Laccadive Sea",
            "Gulf of Aden",
            "Gulf of Oman",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Great Australian Bight",
        ],
        "Arctic Ocean": [
            "Beaufort Sea",
            "Chukchi Sea",
            "Barents Sea",
            "Kara Sea",
            "Laptev Sea",
            "White Sea",
            "The North Western Passages",
            "Amundsen Gulf",
            "Viscount Melville Sound",
        ],
        "SOUTHERN OCEAN": [
            "Ross Sea Eastern Basin",
            "Ross Sea Western Basin",
            "Weddell Sea",
            "Bellingshausen Sea",
            "Amundsen Sea",
            "Scotia Sea",
            "Drake Passage",
        ],
        "Black Sea": None,
        "Mediterranean Sea": [
            "Mediterranean Sea Eastern Basin",
            "Mediterranean Sea Western Basin",
            "Tyrrhenian Sea",
            "Adriatic Sea",
            "Golfe du Lion",
            "Ionian Sea",
            "Strait of Gibraltar",
            "Balearic Sea",
            "Aegean Sea",
        ],
        "Red Sea": None,
        "Caspian Sea": None,
        "Baltic Sea": ["Gulf of Bothnia", "Gulf of Finland"],
    }


def merged_mask(
    basins, ds, lon_name="lon", lat_name="lat", merge_dict=None, verbose=False
):
    """Combine geographical basins (from regionmask) to larger ocean basins.

    Parameters
    ----------
    basins : regionmask.core.regions.Regions object
        Loaded basin data from regionmask, e.g. `import regionmask;basins = regionmask.defined_regions.natural_earth.ocean_basins_50`
    ds : xr.Dataset
        Input dataset on which to construct the mask
    lon_name : str, optional
        Name of the longitude coordinate in `ds`, defaults to `lon`
    lat_name : str, optional
        Name of the latitude coordinate in `ds`, defaults to `lat`
    merge_dict : dict, optional
        dictionary defining new aggregated regions (as keys) and the regions to be merge into that region as as values (list of names).
        Defaults to large scale ocean basins defined by `cmip6_preprocessing.regionmask.default_merge_dict`
    verbose : bool, optional
       Prints more output, e.g. the regions in `basins` that were not used in the merging step. Defaults to False.

    Returns
    -------
    mask : xr.DataArray
        The mask contains ascending numeric value for each key ( merged region) in `merge_dict`.
        When the default is used the numeric values correspond to the following regions:
        * 0: North Atlantic

        * 1: South Atlantic

        * 2: North Pacific

        * 3: South Pacific

        * 4: Maritime Continent

        * 5: Indian Ocean

        * 6: Arctic Ocean

        * 7: Southern Ocean

        * 8: Black Sea

        * 9: Mediterranean Sea

        *10: Red Sea

        *11: Caspian Sea

    """
    mask = basins.mask(ds, lon_name=lon_name, lat_name=lat_name)

    def find_mask_index(name):
        target_value = [
            ri for ri in range(len(basins.regions)) if basins.regions[ri].name == name
        ]
        if len(target_value) > 1:
            warnings.warn(f"Found more than one matching region for {name}")
            return target_value[0]
        elif len(target_value) == 1:
            return target_value[0]
        else:
            return None

    if merge_dict is None:
        merge_dict = _default_merge_dict()

    dict_keys = list(merge_dict.keys())
    number_dict = {k: None for k in dict_keys}
    merged_basins = []
    for ocean, small_basins in merge_dict.items():
        #         ocean_idx = find_mask_index(ocean)
        try:
            ocean_idx = basins.map_keys(ocean)
        except (KeyError):
            # The ocean key is new and cant be found in the previous keys (e.g. for Atlantic full or maritime continent)
            ocean_idx = mask.max().data + 1
        number_dict[ocean] = ocean_idx
        if small_basins:
            for sb in small_basins:
                sb_idx = basins.map_keys(sb)
                # set the index of each small basin to the ocean value
                mask = mask.where(mask != sb_idx, ocean_idx)
                merged_basins.append(sb)

    if verbose:
        remaining_basins = [
            str(basins.regions[ri].name)
            for ri in range(len(basins.regions))
            if (basins.regions[ri].name not in merged_basins)
            and (basins.regions[ri].name not in list(merge_dict.keys()))
        ]
        print(remaining_basins)

    # reset the mask indicies to the order of the passed dictionary keys
    mask_reordered = xr.ones_like(mask.copy()) * np.nan
    for new_idx, k in enumerate(dict_keys):
        old_idx = number_dict[k]
        mask_reordered = mask_reordered.where(mask != old_idx, new_idx)

    return mask_reordered
