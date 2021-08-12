import * as React from "react";
import {
  InputGroup,
  Menu,
  MenuItem,
  Button,
  Spinner,
  Popover,
  Position,
  MenuDivider
} from "@blueprintjs/core";

export interface ISearchbarProps {
  handleFilter: Function;
}

const Searchbar: React.FC<ISearchbarProps> = (props) => {
  const [searchValue, setSearch] = React.useState("");
  const [searching, setSearching] = React.useState(false);
  const maybeSpinner = searching ? <Spinner size={20} /> : undefined;

  const handleSearch = async () => {
    setSearching(true);
    await props.handleFilter(searchValue);
    setSearching(false);
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    window.setTimeout(() => setSearch(value), 10);
  };

  const populateSearch = (searchArgs: string) => {
    searchArgs =
      searchValue !== "" ? searchValue + " " + searchArgs : searchArgs;
    setSearch(searchArgs);
  };

  const filterMenu = (
    <div>
      <Popover
        content={
          <Menu>
            <MenuItem
              text="Filter by name"
              onClick={() => {
                populateSearch("name:");
              }}
            />
            <MenuItem
              text="Filter by label"
              onClick={() => {
                populateSearch("labelKey:");
              }}
            />
            <MenuItem
              text="Filter by api type"
              onClick={() => {
                populateSearch("api:");
              }}
            />
            <MenuItem
              text="Filter by artifact type"
              onClick={() => {
                populateSearch("artifact:");
              }}
            />
            <MenuDivider />
            <MenuItem
              text="Search Syntax"
              icon={"share"}
              target={"_blank"}
              href="https://google.com"
            />
          </Menu>
        }
        position={Position.BOTTOM_LEFT}
      >
        <Button minimal={true} rightIcon="caret-down">
          Filters
        </Button>
      </Popover>
      <Button icon={"search"} minimal={true} onClick={handleSearch} />
    </div>
  );

  return (
    <div style={{ marginBottom: "2rem" }}>
      <InputGroup
        large={true}
        placeholder="Search for a bundle..."
        leftElement={filterMenu}
        rightElement={maybeSpinner}
        value={searchValue}
        onChange={handleSearchChange}
      />
    </div>
  );
};

export default Searchbar;
