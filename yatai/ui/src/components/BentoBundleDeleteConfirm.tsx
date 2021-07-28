import * as React from "react";
import axios from "axios";
import { Redirect } from "react-router-dom";
import { Button, Dialog, Classes, Intent } from "@blueprintjs/core";
import { YataiToaster } from "../utils/Toaster";

export interface IBentoBundleDeleteConfirmationProps {
  name: string;
  value: string;
  isOpen: boolean;
}

const BentoBundleDeleteConfirmation: React.FC<IBentoBundleDeleteConfirmationProps> =
  (props) => {
    const [open, setOpen] = React.useState(false);
    const [isDelete, setDelete] = React.useState(false);
    const bundle: string = `${props.name}:${props.value}`;

    const handleDelete = async (name: string, version: string) => {
      axios
        .post("/api/DeleteBento", { bento_name: name, bento_version: version })
        .then(() => {
          const toastState = {
            message: `${name}:${version} has been deleted.`,
            intent: Intent.SUCCESS,
          };
          setDelete(true);
          YataiToaster.show({ ...toastState });
        })
        .catch((error) => {
          const toastState = {
            message: `${name}:${version} has not been deleted.\nError encountered: ${error}`,
            intent: Intent.DANGER,
          };
          YataiToaster.show({ ...toastState });
        });
    };

    return (
      <div>
        {isDelete ? <Redirect to="/" /> : ""}
        <Button
          outlined={true}
          large={true}
          onClick={() => {
            setOpen(true);
          }}
        >
          Delete
        </Button>

        <Dialog
          icon="info-sign"
          onClose={() => {
            setOpen(false);
          }}
          title="Are you sure?"
          isOpen={open}
        >
          <div className={Classes.DIALOG_BODY}>
            <p>
              This action cannot be undone. This will permanently delete this
              bento bundle and may have unintended consequences.
            </p>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                marginTop: "1.5rem",
              }}
            >
              <Button
                outlined={true}
                large={true}
                text={`Delete ${bundle}`}
                onClick={() => {
                  handleDelete(props.name, props.value);
                }}
              />
            </div>
          </div>
        </Dialog>
      </div>
    );
  };

export default BentoBundleDeleteConfirmation;
